"""Running population statistics shared by the dynamics, actor, and critic."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RunningNorm(nn.Module):
    """Update statistics explicitly; normalize with fixed statistics and ten-sigma clipping."""

    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, values: Tensor) -> None:
        values = values.reshape(-1, self.mean.numel())
        if values.shape[0] == 0:
            return
        if not torch.isfinite(values).all():
            raise FloatingPointError("Normalization data contains non-finite values")
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        count = values.shape[0]
        total = self.count + count
        difference = mean - self.mean
        self.var.copy_(
            (self.var * self.count + variance * count + difference.square() * self.count * count / total) / total
        )
        self.mean.add_(difference * count / total)
        self.count.copy_(total)

    def forward(self, values: Tensor) -> Tensor:
        return ((values - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-10.0, 10.0)

    def unnormalize(self, values: Tensor) -> Tensor:
        return values * (self.var.sqrt() + 1e-6) + self.mean
