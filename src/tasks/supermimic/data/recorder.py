"""Stream the state, noisy history and actions used at each simulator step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

from .replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class ReplayRecorder(RecorderTerm):
    def __init__(self, cfg: ReplayRecorderCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        spec = self.command.model.spec
        history_length, observation_dim = env.observation_manager.group_obs_dim["history"]
        self.buffer = ReplayBuffer(
            env.num_envs,
            cfg.params["capacity"],
            spec.state_dim,
            spec.act_dim,
            spec.xi_dim,
            env.device,
            history_length=history_length,
            observation_dim=observation_dim,
            archive_size=cfg.params.get("archive_size", 100000),
            archive_keep_prob=cfg.params.get("archive_keep_prob", 0.02),
            task=self.command.model,
        )
        self.prev_s = torch.zeros(env.num_envs, spec.state_dim, device=env.device)
        self.prev_xi = torch.zeros(env.num_envs, spec.xi_dim, device=env.device)
        self.prev_history = torch.zeros(env.num_envs, history_length, observation_dim, device=env.device)
        self.clean_action = None
        self._terminal_step: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def command(self):
        return self._env.command_manager.get_term("velocity")

    def _remember(self, ids: torch.Tensor | slice) -> None:
        state = self.command.state
        self.prev_s[ids] = self.command.model.spec.pack(state)[ids]
        self.prev_xi[ids] = state.context[ids]
        self.prev_history[ids] = self._env.obs_buf["history"][ids]

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        if self._terminal_step is None:
            self.buffer.end_episode(env_ids)
            self._remember(env_ids)

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        # Manager resets clear actions and flags before record_post_step runs.
        env = self._env
        self._terminal_step = (
            env.action_manager.action.clone(),
            env.reset_terminated.clone(),
            env.reset_time_outs.clone(),
        )

    def record_post_step(self) -> None:
        env = self._env
        executed, terminated, truncated = self._terminal_step or (
            env.action_manager.action,
            env.reset_terminated,
            env.reset_time_outs,
        )
        clean = self.clean_action if self.clean_action is not None else executed
        self.buffer.add(
            s=self.prev_s,
            a=clean,
            a_exec=executed,
            xi=self.prev_xi,
            r=env.reward_buf,
            terminated=terminated,
            truncated=truncated,
            history=self.prev_history,
        )
        self._remember(slice(None))
        self._terminal_step = None


@dataclass
class ReplayRecorderCfg(RecorderTermCfg):
    func: type = ReplayRecorder
    params: dict = field(default_factory=lambda: {"capacity": 1000, "archive_size": 100000, "archive_keep_prob": 0.02})
