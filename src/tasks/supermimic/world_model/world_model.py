# Copyright (c) 2026-2027 zh
"""Reference autoregressive world model with Go2 state and analytic task heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .dynamics import DynamicsMLP
from .heads import AnalyticHeads


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device):
        super().__init__()
        self.cfg = cfg
        self.spec = heads.spec
        self.dynamics = DynamicsMLP(
            feat_dim=self.spec.feat_dim,
            act_dim=self.spec.act_dim,
            xi_dim=self.spec.xi_dim,
            hidden=cfg.hidden,
            out_dim=self.spec.delta_dim,
        )
        self.heads = heads
        self.to(device)

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor, recorded=None) -> torch.Tensor:
        """Predict physical state, then advance the task's deterministic memory and commands."""
        feat = self.spec.canonical(s, recorded)
        delta = self.dynamics(feat, action, self.spec.normalize_xi(xi))
        state = self.spec.unpack(s, xi)
        physical = self.spec.integrate(state, delta)
        return self.spec.pack(self.heads.advance(state, action, physical))

    def _loss_step(self, s, a, xi, s_true, recorded):
        """One autoregressive step of Eq. 1, including task error coordinates."""
        s_next = self.step(s, a, xi, recorded)
        e = self.spec.error_vector(s, s_next, s_true, recorded, xi)
        return s_next, e * e

    def enable_compile(self, mode: str | None = "default") -> None:
        if not mode:
            self._loss_step_fn = self._loss_step
            return
        # Keep the task's explicit half-to-float terrain rounding under Inductor.
        options = dict(torch._inductor.list_mode_options(mode))
        options["emulate_precision_casts"] = True
        self._loss_step_fn = torch.compile(self._loss_step, options=options, dynamic=False)

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor):
        """Eq. 1: sum error dimensions, then average the batch and autoregressive window."""
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        err_first = None
        err_last = None
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            s_next, sq = step_fn(s, a_exec[j], xi, states[j + 1], states[j])
            total_state = total_state + sq.sum(-1).mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
            s = self.spec.teacher_force(s_next, states[j + 1])
        loss = total_state / k
        return loss, {
            "state_loss": loss.detach(),
            "nse_1step": err_first.detach(),
            "nse_kstep": err_last.detach(),
        }

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Open-loop normalized squared error per step, averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        out = []
        for j in range(k):
            s_next = self.step(s, a_exec[j], xi)
            e = self.spec.error_vector(s, s_next, states[j + 1], states[j], xi)
            out.append((e * e).mean(-1).mean())
            s = s_next
        return torch.stack(out)
