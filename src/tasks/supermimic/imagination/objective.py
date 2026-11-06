"""Survival-weighted policy objective and fixed critic lambda-return targets."""

from __future__ import annotations

import torch

from .rollout import ImagRollout


def policy_objective(ro: ImagRollout, terminal_values: torch.Tensor | None, gamma: float) -> torch.Tensor:
    """Sum discounted rewards and terminal value without horizon normalization."""
    horizon = ro.rewards.shape[0]
    discount = gamma ** torch.arange(horizon, device=ro.rewards.device, dtype=ro.rewards.dtype)
    objective = (discount[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    if terminal_values is not None:
        objective = objective + gamma**horizon * ro.survival[-1] * terminal_values
    return objective.mean()


@torch.no_grad()
def gae_targets(
    rewards: torch.Tensor,
    dhat_next: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
    survival: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use cumulative survival ratios, including the baseline's low-survival clamp."""
    horizon = rewards.shape[0]
    if survival is None:
        survival = torch.cat((torch.ones_like(dhat_next[:1]), (1 - dhat_next).cumprod(0)))
    target = values[-1].clone()
    output = torch.empty_like(rewards)
    for step in reversed(range(horizon)):
        continuation = survival[step + 1] / survival[step].clamp_min(1e-6)
        target = rewards[step] + gamma * continuation * ((1 - lam) * values[step + 1] + lam * target)
        output[step] = target
    return output
