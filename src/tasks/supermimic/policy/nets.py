"""Go2 mixture-of-experts history policy and privileged critic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..envs.mdp.observations import CRITIC_DIM, OBS_DIM, OBS_TERMS
from ..world_model.dynamics import MLP
from ..world_model.state_spec import StateSpec


@dataclass
class PolicyCfg:
    expert_num: int = 8
    expert_hidden: tuple[int, ...] = (512, 256, 256)
    latent_dim: int = 32
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    # Divide angular velocity, gravity, joint position/velocity, actions, commands, phase.
    obs_term_scale: tuple[float, ...] = (2.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0)


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (512, 256, 128)


class HistoryEncoder(nn.Module):
    """Shared features and grouped expert projections, combined by learned gates."""

    def __init__(self, cfg: PolicyCfg, history_length: int):
        super().__init__()
        width = OBS_DIM * history_length
        self.expert_num, self.latent_dim = cfg.expert_num, cfg.latent_dim
        self.backbone = nn.Sequential(
            MLP(width, cfg.expert_hidden[:-1], cfg.expert_num * cfg.expert_hidden[-1], act=nn.ELU), nn.ELU()
        )
        self.experts = nn.Conv1d(
            cfg.expert_num * cfg.expert_hidden[-1], cfg.expert_num * cfg.latent_dim, 1, groups=cfg.expert_num
        )
        self.gates = MLP(width, cfg.expert_hidden[:-1], cfg.expert_num, act=nn.ELU)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        weights = self.gates(history).softmax(dim=-1)
        experts = self.experts(self.backbone(history).unsqueeze(-1)).reshape(-1, self.expert_num, self.latent_dim)
        return F.normalize((weights.unsqueeze(-1) * experts).sum(dim=1), p=2.0, dim=-1)


class Policy(nn.Module):
    """Encode frame-major history and concatenate its latent with the current frame."""

    def __init__(self, cfg: PolicyCfg, history_length: int = 5):
        super().__init__()
        self.cfg = cfg
        self.observation_dim = OBS_DIM
        self.action_dim = StateSpec.act_dim
        self.history_length = history_length
        self.history_order = "frame_major_oldest_first"
        frame_scale = torch.cat([torch.full((n,), 1.0 / s) for (_, n), s in zip(OBS_TERMS, cfg.obs_term_scale)])
        self.register_buffer("scale", frame_scale.repeat(history_length))
        self.encoder = HistoryEncoder(cfg, history_length)
        self.actor = MLP(cfg.latent_dim + OBS_DIM, cfg.actor_hidden, self.action_dim, act=nn.ELU)

    def forward(self, obs_hist: torch.Tensor) -> torch.Tensor:
        history = obs_hist.reshape(-1, self.history_length * self.observation_dim) * self.scale
        return self.actor(torch.cat([self.encoder(history), history[:, -self.observation_dim :]], dim=-1))


class Critic(nn.Module):
    """Value of the task's privileged feature coordinates, with fixed unit scales."""

    def __init__(self, cfg: CriticCfg):
        super().__init__()
        self.mlp = MLP(CRITIC_DIM, cfg.hidden, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat).squeeze(-1)
