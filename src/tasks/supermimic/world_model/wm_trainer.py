"""Fit normalized dynamics on masked, prioritized replay windows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.replay_buffer import ReplayBuffer, WindowBatch
from .world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"
    prioritized_fraction: float = 0.25
    archive_fraction: float = 0.25


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, cfg: WMTrainerCfg):
        self.wm = wm
        self.cfg = cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.valid)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(
                self.cfg.batch_size,
                self.cfg.window,
                prioritized_frac=self.cfg.prioritized_fraction,
                archive_frac=self.cfg.archive_fraction,
            )
            loss, m = self.loss_on(w)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)
            finite = torch.isfinite(gn)
            if bool(finite):
                self.opt.step()
            m["skipped"] = (~finite).to(loss.dtype)
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items() if bool(torch.isfinite(v))}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on recent replay windows."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi, w.valid)
