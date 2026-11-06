"""Normalized autoregressive dynamics with analytic task heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from .dynamics import DynamicsMLP
from .normalization import RunningNorm

if TYPE_CHECKING:
    from .heads import AnalyticHeads


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (1024, 1024, 1024)


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
        self.input_norm = RunningNorm(self.spec.feat_dim + self.spec.act_dim + self.spec.xi_dim)
        self.output_norm = RunningNorm(self.spec.delta_dim)
        self._compiled_supervised = None
        self.to(device)

    def model_input(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor, recorded=None) -> torch.Tensor:
        """Concatenate canonical state, executed action, and simulator context."""
        return torch.cat((self.spec.canonical(s, recorded), action, self.spec.normalize_xi(xi)), -1)

    @torch.no_grad()
    def update_stats(self, s: torch.Tensor, action: torch.Tensor, target: torch.Tensor, xi: torch.Tensor):
        """Update from valid real transitions, keeping contact logits at unit scale."""
        self.input_norm.update(self.model_input(s, action, xi))
        self.output_norm.update(self.spec.model_target(s, target))
        self.output_norm.mean[93:97].zero_()
        self.output_norm.var[93:97].fill_(1.0)

    def _predict_physical(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor, recorded):
        inputs = self.input_norm(self.model_input(s, action, xi, recorded))
        delta = self.output_norm.unnormalize(self.dynamics(inputs))
        physical, _, _ = self.spec.split(s)
        return self.spec.integrate(physical, delta)

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor, recorded=None) -> torch.Tensor:
        """Predict physics before advancing task memory and sampled commands."""
        if recorded is not None and self._compiled_supervised is not None:
            physical = self._compiled_supervised(s, action, xi, recorded)
        else:
            physical = self._predict_physical(s, action, xi, recorded)
        state = self.spec.unpack(s, xi)
        # Supervised command sampling stays eager; imagination compiles its own full step.
        return self.spec.pack(self.heads.advance(state, action, physical))

    def _squared_error(self, current, predicted, target, recorded):
        error = self.spec.error_vector(current, predicted, target, recorded)
        error = error / (self.output_norm.var.sqrt() + 1e-6)
        return error * error

    def _loss_step(self, s, a, xi, s_true, recorded):
        """Predict one supervised step and its normalized coordinate errors."""
        s_next = self.step(s, a, xi, recorded)
        return s_next, self._squared_error(s, s_next, s_true, recorded)

    def enable_compile(self, mode: str | None = "default") -> None:
        """Compile only supervised physical prediction, leaving task sampling outside."""
        if not mode:
            self._compiled_supervised = None
            return
        # Keep the task's explicit half-to-float terrain rounding under Inductor.
        options = dict(torch._inductor.list_mode_options(mode))
        options["emulate_precision_casts"] = True
        self._compiled_supervised = torch.compile(
            self._predict_physical, options=options, fullgraph=True, dynamic=False
        )

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, valid=None):
        """Average per-coordinate errors over valid transitions, including partial windows."""
        k = a_exec.shape[0]
        s = states[0]
        if valid is None:
            valid = torch.ones(a_exec.shape[:2], device=a_exec.device, dtype=a_exec.dtype)
        total_state = 0.0
        err_first = None
        err_last = None
        for j in range(k):
            s_next, sq = self._loss_step(s, a_exec[j], xi, states[j + 1], states[j])
            masked_error = (sq.mean(-1) * valid[j]).sum()
            total_state = total_state + masked_error
            nse = masked_error.detach() / valid[j].sum().clamp_min(1)
            if j == 0:
                err_first = nse
            err_last = nse
            s = self.spec.teacher_force(s_next, states[j + 1])
        loss = total_state / valid.sum().clamp_min(1)
        return loss, {
            "state_loss": loss.detach(),
            "nse_1step": err_first.detach(),
            "nse_kstep": err_last.detach(),
        }

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, valid=None) -> torch.Tensor:
        """Open-loop normalized squared error per step, averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        if valid is None:
            valid = torch.ones(a_exec.shape[:2], device=a_exec.device, dtype=a_exec.dtype)
        out = []
        for j in range(k):
            s_next = self.step(s, a_exec[j], xi)
            sq = self._squared_error(s, s_next, states[j + 1], states[j])
            out.append((sq.mean(-1) * valid[j]).sum() / valid[j].sum().clamp_min(1))
            s = s_next
        return torch.stack(out)
