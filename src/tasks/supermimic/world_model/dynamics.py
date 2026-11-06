"""Shared MLP building block and normalized physical dynamics network."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...],
        out_dim: int,
        act=nn.SiLU,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        last = nn.Linear(d, out_dim)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsMLP(nn.Module):
    """Predict normalized physical residuals from normalized state, action, and context."""

    def __init__(
        self, feat_dim: int = 211, act_dim: int = 12, xi_dim: int = 16, hidden=(1024, 1024, 1024), out_dim: int = 125
    ):
        super().__init__()
        self.mlp = MLP(feat_dim + act_dim + xi_dim, tuple(hidden), out_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)
