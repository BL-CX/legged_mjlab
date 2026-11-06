"""Differentiable rollouts initialized from recorded observation histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .compiled import ImagStep

if TYPE_CHECKING:
    from ..data.replay_buffer import StartBatch
    from ..policy.nets import Policy
    from ..world_model.world_model import WorldModel


@dataclass
class ImagRollout:
    states: torch.Tensor  # [H+1, B, S]
    actions: torch.Tensor  # [H, B, A]
    rewards: torch.Tensor  # [H, B]
    dhat: torch.Tensor  # [H+1, B], with d_0 = 0.
    survival: torch.Tensor  # [H+1, B], cumulative continuation.
    features: torch.Tensor  # [H+1, B, F], privileged critic input.
    reward_terms: dict[str, float]


def imagine(
    wm: WorldModel,
    policy: Policy,
    start: StartBatch,
    horizon: int,
    obs_noise: bool = True,
    step_fn: ImagStep | None = None,
) -> ImagRollout:
    step_fn = step_fn or ImagStep(wm, obs_noise=obs_noise)
    history = list(start.history.unbind(1))
    state = start.state
    survival = torch.ones(state.shape[0], device=state.device)
    states, actions, rewards = [state], [], []
    failures, survivals = [torch.zeros_like(survival)], [survival]
    features, term_sum = [], {}
    for _ in range(horizon):
        with torch.no_grad():
            features.append(wm.heads.privileged(wm.spec.unpack(state, start.xi)))
        action = policy(torch.stack(history, dim=1))
        next_state, reward, terms, failure, frame = step_fn(state, action, start.xi)
        survival = survival * (1 - failure)
        for name, value in terms.items():
            term_sum[name] = term_sum.get(name, 0.0) + value.detach().mean()
        history = history[1:] + [frame]
        state = next_state
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        failures.append(failure)
        survivals.append(survival)
    # Only the terminal critic input keeps its gradient through the imagined state.
    features.append(wm.heads.privileged(wm.spec.unpack(state, start.xi)))
    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(failures),
        survival=torch.stack(survivals),
        features=torch.stack(features),
        reward_terms={name: float(value / horizon) for name, value in term_sum.items()},
    )
