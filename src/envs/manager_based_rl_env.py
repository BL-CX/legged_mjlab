"""MJLab 1.2 environment with optional pre-reset terminal observations.

Capture uses the data format from https://github.com/mujocolab/mjlab/pull/1030.
Initialization, managers and buffers are inherited from the installed MJLab.
"""

from dataclasses import dataclass
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv as MjlabManagerBasedRlEnv
from mjlab.envs import ManagerBasedRlEnvCfg as MjlabManagerBasedRlEnvCfg
from mjlab.envs import types
from mjlab.utils.buffers import CircularBuffer, DelayBuffer


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg(MjlabManagerBasedRlEnvCfg):
  capture_terminal_observations: bool = False
  """Save all pre-reset observation groups for done rows in ``extras["terminal"]``.

  The terminal pass adds no new noise, preserves history/delay and never updates
  their live buffers. It reads the data available before reset without extra
  forward/sense calls; derived quantities retain MJLab 1.2's native update timing.
  """


class ManagerBasedRlEnv(MjlabManagerBasedRlEnv):
  """Native environment with optional terminal observations before auto-reset."""

  cfg: ManagerBasedRlEnvCfg

  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[types.VecEnvObs, dict]:
    del options  # Unused.
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
    if seed is not None:
      self.seed(seed)
    self.extras.pop("terminal", None)
    self._reset_idx(env_ids)
    self.scene.write_data_to_sim()
    self.sim.forward()
    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    return self.obs_buf, self.extras

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    """Follow MJLab 1.2's step order, capturing terminal rows before reset.

    The optional terminal pass leaves the native observation buffers untouched.
    """
    self.extras.pop("terminal", None)
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Check terminations and compute rewards.
    # NOTE: Derived quantities (xpos, xquat, ...) are stale by one physics
    # substep here, as in the native environment.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    # Reset envs that terminated/timed-out and log the episode info.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if self.cfg.capture_terminal_observations and len(reset_env_ids) > 0:
      self._store_terminal_observations(reset_env_ids)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    # Recompute derived quantities from current
    # qpos/qvel for every env. For non-reset envs this resolves the
    # one-substep staleness left by mj_step; for reset envs it picks up
    # the freshly written reset state.
    self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def _store_terminal_observations(self, env_ids: torch.Tensor) -> None:
    self.extras["terminal"] = {
      "env_ids": env_ids.clone(),
      "observations": self._compute_terminal_observations(env_ids),
    }

  @torch.no_grad()
  def _compute_terminal_observations(self, env_ids: torch.Tensor) -> types.VecEnvObs:
    """Read every group without new noise or manager-cache/buffer updates.

    Existing delayed/history frames retain their original values. Observation
    terms are expected to be read-only, as in the native observation manager.
    """
    manager = self.observation_manager
    observations: types.VecEnvObs = {}
    for group, term_names in manager.active_terms.items():
      group_cfg = manager.cfg[group]
      terms: dict[str, torch.Tensor] = {}
      for name in term_names:
        cfg = manager.get_term_cfg(group, name)
        obs = cfg.func(self, **cfg.params).clone()
        if cfg.clip:
          obs = obs.clamp_(min=cfg.clip[0], max=cfg.clip[1])
        if cfg.scale is not None:
          obs = obs.mul_(cfg.scale)
        if group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
          obs = manager._check_and_handle_nans(
            obs, context=f"{group}/{name}", policy=group_cfg.nan_policy
          )
        if cfg.delay_max_lag > 0:
          obs = _peek_delay(manager._group_obs_term_delay_buffer[group][name], obs)
        if cfg.history_length > 0:
          obs = _peek_history(manager._group_obs_term_history_buffer[group][name], obs)
          if cfg.flatten_history_dim:
            obs = obs.reshape(self.num_envs, -1)
        terms[name] = obs

      if manager.group_obs_concatenate[group]:
        value = torch.cat(
          list(terms.values()), dim=manager._group_obs_concatenate_dim[group]
        )
        if not group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
          value = manager._check_and_handle_nans(
            value, context=group, policy=group_cfg.nan_policy
          )
        observations[group] = value[env_ids].clone()
      else:
        if not group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
          terms = {
            name: manager._check_and_handle_nans(
              obs, context=f"{group}/{name}", policy=group_cfg.nan_policy
            )
            for name, obs in terms.items()
          }
        observations[group] = {name: obs[env_ids].clone() for name, obs in terms.items()}
    return observations


def _peek_history(buffer: CircularBuffer, data: torch.Tensor) -> torch.Tensor:
  """Build the next history window without appending to the native buffer."""
  data = data.to(buffer.device)
  if not buffer.is_initialized:
    return data.unsqueeze(1).expand(-1, buffer.max_length, *data.shape[1:]).clone()
  history = torch.cat((buffer.buffer[:, 1:], data.unsqueeze(1)), dim=1)
  empty = buffer.current_length == 0
  history[empty] = data[empty].unsqueeze(1)
  return history


def _peek_delay(buffer: DelayBuffer, data: torch.Tensor) -> torch.Tensor:
  """Use the current lag, without appending, resampling or incrementing counters."""
  data = data.to(buffer.device)
  history = buffer._buffer
  if not history.is_initialized:
    return data.clone()
  length = (history.current_length + 1).clamp_max(history.max_length)
  lags = torch.minimum(buffer.current_lags, length - 1).clamp_min(0)
  previous = history[(lags - 1).clamp_min(0)]
  current = (lags == 0).reshape(-1, *([1] * (data.ndim - 1)))
  return torch.where(current, data, previous)
