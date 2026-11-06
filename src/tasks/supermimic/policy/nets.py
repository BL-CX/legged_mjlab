"""Mixture-of-experts history policy and privileged critic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..envs.mdp.observations import CRITIC_DIM, OBS_DIM
from ..world_model.dynamics import MLP
from ..world_model.normalization import RunningNorm
from ..world_model.state_spec import StateSpec


@dataclass
class PolicyCfg:
    expert_num: int = 8
    expert_hidden: tuple[int, ...] = (512, 256, 256)
    latent_dim: int = 32
    actor_hidden: tuple[int, ...] = (512, 256, 128)


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
        self.normalizer = RunningNorm(OBS_DIM * history_length)
        self.frame_normalizer = RunningNorm(OBS_DIM)
        self.encoder = HistoryEncoder(cfg, history_length)
        self.actor = MLP(cfg.latent_dim + OBS_DIM, cfg.actor_hidden, self.action_dim, act=nn.ELU)

    @torch.no_grad()
    def update_statistics(self, history: torch.Tensor) -> None:
        """Update history and current-frame statistics once per real collection step."""
        history = history.reshape(-1, self.history_length, self.observation_dim)
        self.normalizer.update(history.reshape(-1, self.history_length * self.observation_dim))
        self.frame_normalizer.update(history[:, -1])

    def forward(self, obs_hist: torch.Tensor) -> torch.Tensor:
        history = obs_hist.reshape(-1, self.history_length * self.observation_dim)
        latent = self.encoder(self.normalizer(history))
        current = self.frame_normalizer(history[:, -self.observation_dim :])
        return self.actor(torch.cat([latent, current], dim=-1))


class Critic(nn.Module):
    """Normalize privileged features before an ELU value network."""

    def __init__(self, cfg: CriticCfg):
        super().__init__()
        self.normalizer = RunningNorm(CRITIC_DIM)
        self.mlp = MLP(CRITIC_DIM, cfg.hidden, 1, act=nn.ELU)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.normalizer(feat)).squeeze(-1)
