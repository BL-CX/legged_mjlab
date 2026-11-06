"""Improve the policy through imagined trajectories and fit the privileged critic."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.replay_buffer import ReplayBuffer
from ..policy.nets import Critic, Policy
from ..world_model.world_model import WorldModel
from .compiled import ImagStep
from .objective import gae_targets, policy_objective
from .rollout import ImagRollout, imagine


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, cfg: PolicyTrainerCfg):
        self.wm, self.policy, self.critic, self.cfg = wm, policy, critic, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        self.step_fn = ImagStep(wm, compile_mode=cfg.compile, obs_noise=cfg.obs_noise)
        self.nonfinite_steps = 0

    def set_lr(self, lr: float, actor_multiplier: float = 1.0) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr * actor_multiplier
        for g in self.critic_opt.param_groups:
            g["lr"] = lr * self.cfg.critic_lr / self.cfg.lr

    def update(self, buffer: ReplayBuffer, *, freeze_actor=False) -> dict[str, float]:
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size)
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        ro = imagine(
            self.wm,
            self.policy,
            start,
            cfg.horizon,
            obs_noise=cfg.obs_noise,
            step_fn=self.step_fn,
        )
        metrics = self._update_actor(ro, freeze_actor=freeze_actor)
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)

        if cfg.use_critic:
            metrics.update(self._update_critic(ro))
        metrics["nonfinite_steps"] = float(self.nonfinite_steps)
        return {key: value for key, value in metrics.items() if torch.isfinite(torch.tensor(value))}

    def _update_actor(self, ro: ImagRollout, *, freeze_actor: bool) -> dict[str, float]:
        """Backpropagate the complete horizon objective through the frozen model and critic."""
        cfg = self.cfg
        v_H = self.critic(ro.features[-1]) if cfg.use_critic else None
        J = policy_objective(ro, v_H, cfg.gamma)
        loss = -J
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        finite = bool(torch.isfinite(gn)) and bool(torch.isfinite(J))
        if finite and not freeze_actor:
            self.opt.step()
        if not finite:
            self.nonfinite_steps += 1

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(gn),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
            "control_skipped": float(not finite),
        }
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v
        return metrics

    def _update_critic(self, ro: ImagRollout) -> dict[str, float]:
        """Fit fixed lambda returns with environment-major minibatches."""
        cfg = self.cfg
        with torch.no_grad():
            features = ro.features.detach()
            H1, n, _ = features.shape
            values = self.critic(features.reshape(H1 * n, -1)).reshape(H1, n)
            targets = gae_targets(
                ro.rewards.detach(),
                ro.dhat[1:].detach(),
                values,
                cfg.gamma,
                cfg.gae_lambda,
                ro.survival.detach(),
            )
            # Flatten environments first to preserve the baseline's seeded minibatch order.
            inputs = features[:-1].transpose(0, 1).reshape(-1, features.shape[-1])
            labels = targets.transpose(0, 1).reshape(-1)
            self.critic.normalizer.update(inputs)

        count = labels.shape[0]
        width = count // cfg.critic_minibatches
        losses, skipped = [], 0
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(count, device=labels.device)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * width : (i + 1) * width]
                loss = torch.nn.functional.mse_loss(self.critic(inputs[idx]), labels[idx])
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                if bool(torch.isfinite(grad_norm)):
                    self.critic_opt.step()
                else:
                    skipped += 1
                    self.nonfinite_steps += 1
                losses.append(loss.detach())
        return {
            "critic_loss": float(torch.stack(losses).mean()),
            "critic_skipped": float(skipped),
            "critic_value_mean": float(values.mean()),
            "critic_target_mean": float(targets.mean()),
        }
