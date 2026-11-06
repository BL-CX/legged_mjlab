"""Control-step replay with recorded observation histories and rare-event windows.

Rows hold the state and noisy history used to choose an action. The next row is
the transition target; reset rows remain in storage and are excluded by ``valid``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import torch


@dataclass
class WindowBatch:
    states: torch.Tensor  # (k+1, n, S)
    a_exec: torch.Tensor  # (k, n, A)
    a_clean: torch.Tensor  # (k, n, A)
    xi: torch.Tensor  # (n, X), context at the window start
    reward: torch.Tensor  # (k, n)
    done: torch.Tensor  # (k, n), physical termination labels
    valid: torch.Tensor  # (k, n), false from the first reset onward

    def index(self, ids: torch.Tensor | slice) -> WindowBatch:
        return WindowBatch(
            **{
                item.name: getattr(self, item.name)[ids] if item.name == "xi" else getattr(self, item.name)[:, ids]
                for item in fields(self)
            }
        )

    @staticmethod
    def concatenate(parts: Sequence[WindowBatch]) -> WindowBatch:
        return WindowBatch(
            **{
                item.name: torch.cat([getattr(part, item.name) for part in parts], dim=0 if item.name == "xi" else 1)
                for item in fields(WindowBatch)
            }
        )


@dataclass
class StartBatch:
    state: torch.Tensor
    xi: torch.Tensor
    history: torch.Tensor  # (n, hist, D), recorded noisy observations


class WindowArchive:
    """Lazily allocate complete windows so replay overwrites cannot invalidate them."""

    def __init__(self, capacity: int):
        self.capacity, self.size, self.clock = capacity, 0, 0
        self.storage: WindowBatch | None = None

    @torch.no_grad()
    def add(self, window: WindowBatch) -> None:
        count = min(window.a_exec.shape[1], self.capacity)
        if not count:
            return
        window = window.index(slice(0, count))
        if self.storage is None:
            storage = {}
            for item in fields(window):
                value = getattr(window, item.name)
                shape = list(value.shape)
                shape[0 if item.name == "xi" else 1] = self.capacity
                storage[item.name] = value.new_zeros(shape)
            self.storage = WindowBatch(**storage)
        ids = (self.clock + torch.arange(count, device=window.states.device)) % self.capacity
        for item in fields(WindowBatch):
            target, source = getattr(self.storage, item.name), getattr(window, item.name)
            target.index_copy_(0 if item.name == "xi" else 1, ids, source)
        self.clock += count
        self.size = min(self.size + count, self.capacity)

    def sample(self, count: int) -> WindowBatch:
        ids = torch.randint(self.size, (count,), device=self.storage.states.device)
        return self.storage.index(ids)


class ReplayBuffer:
    """Time-major ring replay; collected histories are stored in float16."""

    def __init__(
        self,
        num_envs,
        capacity,
        state_dim,
        act_dim,
        xi_dim,
        device,
        *,
        history_length=6,
        observation_dim=47,
        archive_size=100000,
        archive_keep_prob=0.02,
        task=None,
    ):
        self.E, self.T = num_envs, capacity
        self.device = torch.device(device)
        self.size = self.clock = 0
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.s = torch.zeros(capacity, num_envs, state_dim, device=device)
        self.xi = torch.zeros(capacity, num_envs, xi_dim, device=device)
        self.a = torch.zeros(capacity, num_envs, act_dim, device=device)
        self.a_exec = torch.zeros_like(self.a)
        self.r = torch.zeros(capacity, num_envs, device=device)
        self.done = torch.zeros(capacity, num_envs, dtype=torch.bool, device=device)
        self.terminated = torch.zeros_like(self.done)
        self.histories = torch.zeros(
            capacity, num_envs, history_length, observation_dim, device=device, dtype=torch.float16
        )
        self.archive = WindowArchive(archive_size)
        self.archive_keep_prob, self.task = archive_keep_prob, task

    @torch.no_grad()
    def add(self, *, s, a, a_exec, xi, r, terminated, truncated, history):
        slot = self.clock % self.T
        for target, value in (
            (self.s, s),
            (self.a, a),
            (self.a_exec, a_exec),
            (self.xi, xi),
            (self.r, r),
            (self.terminated, terminated),
            (self.histories, history),
        ):
            target[slot].copy_(value)
        self.done[slot].copy_(terminated | truncated)
        self.clock += 1
        self.size = min(self.size + 1, self.T)

    def end_episode(self, env_ids: torch.Tensor) -> None:
        # An explicit partial reset also cuts the preceding transition window.
        if self.size:
            self.done[(self.clock - 1) % self.T, env_ids] = True

    def ready(self, horizon: int) -> bool:
        return self.size > horizon + 1

    def _uniform_windows(self, count: int, horizon: int) -> WindowBatch:
        if not self.ready(horizon):
            raise ValueError("Replay has insufficient state slots for sampling")
        rows = torch.randint(self.E, (count,), device=self.device)
        logical = torch.randint(0, self.size - horizon, (count,), device=self.device)
        starts = (self.clock - self.size + logical) % self.T
        slots = (starts[None] + torch.arange(horizon + 1, device=self.device)[:, None]) % self.T
        done = self.done[slots[:-1], rows[None]]
        return WindowBatch(
            states=self.s[slots, rows[None]],
            a_exec=self.a_exec[slots[:-1], rows[None]],
            a_clean=self.a[slots[:-1], rows[None]],
            xi=self.xi[starts, rows],
            reward=self.r[slots[:-1], rows[None]],
            done=self.terminated[slots[:-1], rows[None]].float(),
            valid=(~done).cumprod(0).bool(),
        )

    @torch.no_grad()
    def sample_windows(
        self, n: int, k: int, prioritized_frac: float = 0.25, archive_frac: float = 0.25
    ) -> WindowBatch:
        prioritized = int(n * prioritized_frac)
        archived = int(n * archive_frac) if self.archive.size else 0
        uniform = n - prioritized - archived
        parts = [self._uniform_windows(uniform, k)] if uniform else []
        candidates = self._uniform_windows(4 * max(prioritized, 1), k)
        scores = self.task.replay_priority(candidates)
        worthy = self.task.archive_worthy(candidates)
        keep = worthy & (torch.rand_like(scores) < self.archive_keep_prob)
        self.archive.add(candidates.index(keep))
        chosen = (scores + 1e-3 * torch.rand_like(scores)).topk(prioritized).indices
        if prioritized:
            parts.append(candidates.index(chosen))
        if archived:
            parts.append(self.archive.sample(archived))
        return WindowBatch.concatenate(parts) if len(parts) > 1 else parts[0]

    @torch.no_grad()
    def sample_starts(self, n: int) -> StartBatch:
        rows = torch.randint(self.E, (n,), device=self.device)
        logical = torch.randint(self.size, (n,), device=self.device)
        slots = (self.clock - self.size + logical) % self.T
        return StartBatch(self.s[slots, rows], self.xi[slots, rows], self.histories[slots, rows].float())
