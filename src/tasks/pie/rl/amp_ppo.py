# Copyright (c) 2026-2027 zh
"""PIE PPO with an independent AMP discriminator and motion prior."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO

from .amp.amp_discriminator import AMPDiscriminator
from .amp.amp_motion_loader import AMPMotionLoader
from .amp.replay_buffer import ReplayBuffer
from .amp.normalizer import Normalizer
from .ppo import PIEPPO


class PIEAMPPPO(PIEPPO):
    """Train an AMP discriminator alongside recurrent PIE updates."""

    # ---------- init ----------

    def __init__(
        self,
        *args,
        amp_motion_files: list[str],
        reorder_motion_data: bool = False,
        amp_reward_coef: float = 0.15,
        amp_task_reward_lerp: float = 0.9,
        amp_num_preload_transitions: int = 200_000,
        amp_discr_hidden_dims: tuple[int, ...] = (1024, 512),
        amp_replay_buffer_size: int = 1_000_000,
        amp_learning_rate: float = 1.0e-4,
        amp_num_updates: int = 4,
        amp_mini_batch_size: int = 1024,
        amp_grad_penalty_coef: float = 10.0,
        amp_max_grad_norm: float = 1.0,
        amp_time_between_frames: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if "amp" not in self.storage.observations:
            raise ValueError("PIE AMP requires an independent 'amp' observation group")
        if self.storage.observations["amp"].ndim != 3:
            raise ValueError("PIE AMP observations must have shape [num_envs, features]")
        if not amp_motion_files:
            raise ValueError("PIE AMP requires at least one motion file")
        if min(amp_num_preload_transitions, amp_num_updates, amp_mini_batch_size) < 1:
            raise ValueError("AMP preload count, update count and batch size must be positive")
        if amp_replay_buffer_size < self.storage.num_envs:
            raise ValueError("AMP replay capacity must hold at least one environment step")
        if not amp_discr_hidden_dims or min(amp_discr_hidden_dims) < 1:
            raise ValueError("AMP discriminator hidden dimensions must be positive")
        if min(amp_learning_rate, amp_max_grad_norm, amp_time_between_frames) <= 0:
            raise ValueError("AMP learning rate, gradient limit and transition dt must be positive")
        if min(amp_reward_coef, amp_grad_penalty_coef) < 0:
            raise ValueError("AMP reward and gradient penalty coefficients must be nonnegative")

        # Preloaded transitions provide the 30-dimensional expert state.
        self.amp_data = AMPMotionLoader(
            self.device,
            time_between_frames=amp_time_between_frames,
            preload_transitions=True,
            num_preload_transitions=amp_num_preload_transitions,
            motion_files=amp_motion_files,
            reorder=reorder_motion_data,
        )
        self.amp_obs_dim = self.amp_data.observation_dim
        if self.storage.observations["amp"].shape[-1] != self.amp_obs_dim:
            raise ValueError("Environment and expert AMP observation dimensions must match")
        self.amp_normalizer = Normalizer(self.amp_obs_dim)
        self.discriminator = AMPDiscriminator(
            self.amp_obs_dim * 2,
            amp_reward_coef,
            amp_discr_hidden_dims,
            self.device,
            amp_task_reward_lerp=amp_task_reward_lerp,
        )
        self.amp_storage = ReplayBuffer(self.amp_obs_dim, amp_replay_buffer_size, self.device)
        self.amp_optimizer = torch.optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 1.0e-3},
                {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 0.1},
            ],
            lr=amp_learning_rate,
        )
        self.amp_num_updates = amp_num_updates
        self.amp_mini_batch_size = amp_mini_batch_size
        self.amp_grad_penalty_coef = amp_grad_penalty_coef
        self.amp_max_grad_norm = amp_max_grad_norm

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        """Construct the algorithm using the environment control interval."""
        cfg["algorithm"]["amp_time_between_frames"] = env.unwrapped.step_dt
        return PPO.construct_algorithm(obs, env, cfg, device)

    # ---------- rollout ----------

    def act(self, obs):
        actions = super().act(obs)
        # Snapshot the state before the environment reuses its observation buffers.
        self.transition.observations["amp"] = obs["amp"].detach().clone()
        return actions

    @torch.no_grad()
    def process_env_step(self, obs, rewards, dones, extras) -> None:
        if self.transition.observations is None:
            raise RuntimeError("PIEAMPPPO.process_env_step() was called before act()")
        current_amp = self.transition.observations["amp"]
        next_amp = obs["amp"].detach().clone()
        # Approximate terminal states with the pre-step state after auto-reset.
        done_mask = dones.reshape(-1).bool()
        next_amp[done_mask] = current_amp[done_mask]
        mixed_rewards, style_rewards = self.discriminator.predict_amp_reward(
            current_amp, next_amp, rewards, normalizer=self.amp_normalizer
        )
        self.amp_storage.insert(current_amp, next_amp)

        # Report mixed rewards separately from the runner's task reward metrics.
        log_key = "episode" if "episode" in extras else "log"
        extras[log_key] = {
            **extras.get(log_key, {}),
            "AMP/task_reward": rewards.mean(),
            "AMP/style_reward": style_rewards.mean(),
            "AMP/mixed_reward": mixed_rewards.mean(),
        }
        super().process_env_step(obs, mixed_rewards, dones, extras)

    # ---------- update ----------

    def update(self) -> dict[str, float]:
        losses = super().update()
        losses.update(self._update_discriminator())
        return losses

    def _update_discriminator(self) -> dict[str, float]:
        if self.amp_storage.num_samples == 0:
            raise RuntimeError("AMP discriminator update requires collected transitions")
        means = torch.zeros(4, device=self.device)
        policy_batches = self.amp_storage.feed_forward_generator(
            self.amp_num_updates, self.amp_mini_batch_size
        )
        expert_batches = self.amp_data.feed_forward_generator(
            self.amp_num_updates, self.amp_mini_batch_size
        )
        for (policy, policy_next), (expert, expert_next) in zip(policy_batches, expert_batches):
            with torch.no_grad():
                policy_norm = self.amp_normalizer.normalize_torch(policy, self.device)
                policy_next_norm = self.amp_normalizer.normalize_torch(policy_next, self.device)
                expert_norm = self.amp_normalizer.normalize_torch(expert, self.device)
                expert_next_norm = self.amp_normalizer.normalize_torch(expert_next, self.device)
            policy_pred = self.discriminator(torch.cat((policy_norm, policy_next_norm), dim=-1))
            expert_pred = self.discriminator(torch.cat((expert_norm, expert_next_norm), dim=-1))
            discriminator_loss = 0.5 * (
                (expert_pred - 1.0).square().mean() + (policy_pred + 1.0).square().mean()
            )
            grad_penalty = self.discriminator.compute_grad_pen(
                expert_norm, expert_next_norm, lambda_=self.amp_grad_penalty_coef
            )
            self.amp_optimizer.zero_grad()
            (discriminator_loss + grad_penalty).backward()
            if self.is_multi_gpu:
                for parameter in self.discriminator.parameters():
                    if parameter.grad is not None:
                        torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                        parameter.grad.div_(self.gpu_world_size)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.amp_max_grad_norm)
            self.amp_optimizer.step()
            means += torch.stack(
                (discriminator_loss.detach(), grad_penalty.detach(),
                 policy_pred.detach().mean(), expert_pred.detach().mean())
            )
            self._update_amp_normalizer(torch.cat((policy, policy_next, expert, expert_next)))

        values = (means / self.amp_num_updates).tolist()
        return dict(zip(
            ("amp_discriminator", "amp_grad_penalty", "amp_policy_prediction", "amp_expert_prediction"),
            values,
        ))

    def _update_amp_normalizer(self, raw_states: torch.Tensor) -> None:
        """Update raw-state statistics and merge moments across ranks."""
        if not self.is_multi_gpu:
            self.amp_normalizer.update(raw_states.detach().cpu().numpy())
            return
        raw_states = raw_states.detach().double()
        moments = torch.cat((
            raw_states.sum(dim=0), raw_states.square().sum(dim=0),
            raw_states.new_tensor([raw_states.shape[0]]),
        ))
        torch.distributed.all_reduce(moments, op=torch.distributed.ReduceOp.SUM)
        count = moments[-1].item()
        mean = moments[:self.amp_obs_dim] / count
        var = (moments[self.amp_obs_dim:-1] / count - mean.square()).clamp_min(0.0)
        self.amp_normalizer.update_from_moments(mean.cpu().numpy(), var.cpu().numpy(), count)

    # ---------- mode and distributed ----------

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        for tensor in self.discriminator.state_dict().values():
            torch.distributed.broadcast(tensor, src=0)
