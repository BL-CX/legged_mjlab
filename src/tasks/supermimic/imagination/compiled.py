# Copyright (c) 2026-2027 zh
"""One compiled imagined step: dynamics, Go2 reward, termination, and clean observation."""

from __future__ import annotations

import torch

from ..world_model.world_model import WorldModel


class ImagStep:
    def __init__(self, wm: WorldModel, compile_mode: str | None = None):
        self.wm = wm
        self._fn = self._step
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)

    def _step(self, s, a, xi):
        s_next = self.wm.step(s, a, xi)
        state = self.wm.spec.unpack(s, xi)
        next_state = self.wm.spec.unpack(s_next, xi)
        task = self.wm.heads
        d = task.soft_termination(next_state)
        terms = task.reward_terms(state, a, next_state, d)
        r = task.dt * sum(getattr(task.cfg.reward.scales, name) * value for name, value in terms.items())
        obs_clean = task.observe(next_state, noisy=False)
        return s_next, r, terms, d, obs_clean

    def __call__(self, s, a, xi):
        return self._fn(s, a, xi)
