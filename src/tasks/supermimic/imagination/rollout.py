"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.replay_buffer import StartBatch
from ..policy.nets import Policy
from ..world_model.world_model import WorldModel
from .compiled import ImagStep


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, S)
    actions: torch.Tensor  # (H, n, 12)
    rewards: torch.Tensor  # (H, n) r_{t+1}
    dhat: torch.Tensor  # (H+1, n) predicted termination probability, d_0 = 0
    survival: torch.Tensor  # (H+1, n) c_t
    features: torch.Tensor  # (H+1, n, privileged_dim)
    reward_terms: dict  # Mean of each raw task reward term over the rollout.


def imagine(
    wm: WorldModel,
    policy: Policy,
    start: StartBatch,
    horizon: int,
    obs_noise: bool = True,
    step_fn: ImagStep | None = None,
) -> ImagRollout:
    device = start.state.device
    n = start.state.shape[0]
    step_fn = step_fn or ImagStep(wm)
    noise_half = wm.heads.noise_amplitude

    def add_noise(o):
        if obs_noise and wm.heads.cfg.noise.add_noise:
            return o + (torch.rand_like(o) * 2 - 1) * noise_half
        return o

    # Reconstruct clean logged observations; re-sample noise for every history frame.
    hist = [
        add_noise(wm.heads.observe(wm.spec.unpack(start.hist_states[i], start.xi), noisy=False))
        for i in range(start.hist_states.shape[0])
    ]
    s = start.state
    c = torch.ones(n, device=device)
    states = [s]
    actions, rewards = [], []
    dhats, survivals = [torch.zeros(n, device=device)], [c]
    term_sum = {}

    for _ in range(horizon):
        obs_flat = torch.stack(hist, dim=1).reshape(n, -1)
        a = policy(obs_flat)
        s_next, r, terms, d, obs_clean = step_fn(s, a, start.xi)
        c = c * (1.0 - d)
        for name, value in terms.items():
            term_sum[name] = term_sum.get(name, 0.0) + value.detach().mean()
        hist = hist[1:] + [add_noise(obs_clean)]
        s = s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)

    states = torch.stack(states)
    xi = start.xi[None].expand(horizon + 1, n, -1)
    features = wm.heads.privileged(
        wm.spec.unpack(states.reshape((horizon + 1) * n, -1), xi.reshape((horizon + 1) * n, -1))
    )
    return ImagRollout(
        states=states,
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        features=features.reshape(horizon + 1, n, -1),
        reward_terms={name: float(value / horizon) for name, value in term_sum.items()},
    )
