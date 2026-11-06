"""Training panels, episode statistics, and TensorBoard reporting."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.utils.os import dump_yaml
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from .config import TrainConfig


class TrainingLogger:
    """Write every iteration; print the first, interval, and final training panels."""

    def __init__(self, cfg: TrainConfig, *, num_envs: int, device: torch.device, step_dt: float):
        self.cfg = cfg
        self.num_envs, self.device, self.step_dt = num_envs, device, step_dt
        if cfg.agent.schedule.log_interval < 1:
            raise ValueError("Logging interval must be positive")
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.log_dir = Path(cfg.log_root).expanduser().resolve() / cfg.run_name / stamp
        self.log_dir.mkdir(parents=True, exist_ok=False)
        dump_yaml(self.log_dir / "params" / "env.yaml", asdict(cfg.env))
        dump_yaml(self.log_dir / "params" / "agent.yaml", asdict(cfg.agent))
        dump_yaml(
            self.log_dir / "params" / "run.yaml",
            {
                "run_name": cfg.run_name,
                "device": str(device),
                "num_envs": num_envs,
                "seed": cfg.env.seed,
                "control_dt_seconds": step_dt,
                "resume": str(cfg.resume) if cfg.resume is not None else None,
                "started_at": datetime.now().astimezone().isoformat(),
            },
        )
        self._file = (self.log_dir / "train.log").open("x", encoding="utf-8", buffering=1)
        self.writer = SummaryWriter(str(self.log_dir / "tb"), flush_secs=10)
        self.episode_returns = torch.zeros(num_envs, device=device)
        self.episode_lengths = torch.zeros(num_envs, device=device)
        self.completed_episodes: deque[tuple[float, float]] = deque(maxlen=100)
        self._last_panel: tuple[dict[str, float], str] | None = None
        self._printed_iteration: int | None = None

    @torch.no_grad()
    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Accumulate rewards and control-step lengths of completed episodes."""
        self.episode_returns += rewards
        self.episode_lengths += 1
        if dones.any():
            self.completed_episodes.extend(
                zip(self.episode_returns[dones].tolist(), self.episode_lengths[dones].tolist())
            )
            self.episode_returns[dones] = 0
            self.episode_lengths[dones] = 0

    def episode_statistics(self) -> dict[str, float]:
        """Mean return and control-step length over the last 100 completed episodes."""
        if not self.completed_episodes:
            return {}
        count = len(self.completed_episodes)
        return {
            "train/episode_return": sum(item[0] for item in self.completed_episodes) / count,
            "train/episode_len": sum(item[1] for item in self.completed_episodes) / count,
        }

    def begin(self, iteration: int, real_steps: int) -> None:
        """Measure the current learning session after replay prefill, including resume."""
        self._start_iteration = iteration
        self._last_steps = real_steps
        self._started = self._last_time = time.perf_counter()

    def message(self, message: str) -> None:
        print(message, flush=True)
        self._file.write(message + "\n")

    def record(self, metrics: dict[str, float], *, phase: str) -> None:
        """Report actual transition throughput and retain raw metric names."""
        values = {key: float(value) for key, value in metrics.items() if math.isfinite(value)}
        iteration, real_steps = int(values["iteration"]), int(values["real_steps"])
        now = time.perf_counter()
        elapsed, duration = now - self._started, now - self._last_time
        transitions = real_steps - self._last_steps
        self._last_time, self._last_steps = now, real_steps
        values.update(
            {
                "perf/total_fps": transitions / duration,
                "perf/collection_fps": transitions / values["time/collection_seconds"],
                "time/iteration_seconds": duration,
                "time/session_elapsed_seconds": elapsed,
                "time/eta_seconds": elapsed
                / (iteration - self._start_iteration)
                * max(0, self.cfg.agent.schedule.total_iterations - iteration),
            }
        )
        for key, value in values.items():
            self.writer.add_scalar(key, value, iteration)
        self._last_panel = values, phase
        if (
            iteration == self._start_iteration + 1
            or iteration % self.cfg.agent.schedule.log_interval == 0
            or iteration == self.cfg.agent.schedule.total_iterations
        ):
            self._print_panel(values, phase)

    def _print_panel(self, values: dict[str, float], phase: str) -> None:
        iteration = int(values["iteration"])
        title = f" Learning iteration {iteration}/{self.cfg.agent.schedule.total_iterations} "
        lines = ["#" * 80, title.center(80), ""]

        def row(label: str, value: str) -> None:
            lines.append(f"{label + ':':>40} {value}")

        row("Run name", self.cfg.run_name)
        row("Device / environments", f"{self.device} / {self.num_envs}")
        row("Training phase", phase)
        for label, key, fmt in (
            ("Total real steps", "real_steps", ".0f"),
            ("Steps per second", "perf/total_fps", ".0f"),
            ("Collection steps per second", "perf/collection_fps", ".0f"),
            ("Collection time (s)", "time/collection_seconds", ".3f"),
            ("World model time (s)", "time/wm_seconds", ".3f"),
            ("Policy time (s)", "time/policy_seconds", ".3f"),
            ("Mean world model loss", "wm/loss", ".4f"),
            ("One-step normalized error", "wm/nse_1step", ".4f"),
            ("Window-end normalized error", "wm/nse_kstep", ".4f"),
            ("Imagined objective", "policy/imag_objective", ".4f"),
            ("Mean value loss", "policy/critic_loss", ".4f"),
            ("Imagined horizon survival", "policy/imag_survival_H", ".4f"),
            ("World model learning rate", "lr", ".2e"),
            ("Actor learning rate", "actor_lr", ".2e"),
            ("Value learning rate", "critic_lr", ".2e"),
            ("Mean reward", "train/episode_return", ".2f"),
            ("Mean episode length (steps)", "train/episode_len", ".2f"),
            ("Failures", "failures", ".0f"),
            ("Timeouts", "timeouts", ".0f"),
            ("Replay transitions", "replay_rows", ".0f"),
            ("Archived windows", "archive_size", ".0f"),
            ("Terrain level", "task/terrain_level", ".2f"),
            ("Action RMS", "task/action_rms", ".4f"),
            ("WM skipped update fraction", "wm/skipped", ".3f"),
            ("Actor skipped update", "policy/control_skipped", ".0f"),
            ("Value skipped updates", "policy/critic_skipped", ".0f"),
            ("Guard falls per reset step", "guard/falls_per_reset_step", ".2f"),
        ):
            if key in values:
                row(label, format(values[key], fmt))
        if "train/episode_len" in values:
            row("Mean episode length (s)", f"{values['train/episode_len'] * self.step_dt:.2f}")
        for label, key in (
            ("Planar velocity error RMS (m/s)", "task/tracking_linear_mse"),
            ("Yaw rate error RMS (rad/s)", "task/tracking_angular_mse"),
        ):
            if key in values:
                row(label, f"{math.sqrt(max(0, values[key])):.4f}")
        # Reward rates are instantaneous weighted terms, distinct from native episode summaries.
        for key in sorted(values):
            if key.startswith(("Reward_Rate/", "Episode_Termination/")):
                row(key, f"{values[key]:.4f}")
        lines.append("-" * 80)
        row("Iteration time (s)", f"{values['time/iteration_seconds']:.3f}")
        row("Time elapsed", str(timedelta(seconds=int(values["time/session_elapsed_seconds"]))))
        row("ETA", str(timedelta(seconds=int(values["time/eta_seconds"]))))
        self.message("\n".join(lines) + "\n")
        self._printed_iteration = iteration

    def close(self) -> None:
        if not self._file.closed:
            if self._last_panel is not None and self._printed_iteration != int(self._last_panel[0]["iteration"]):
                self._print_panel(*self._last_panel)
            self.writer.close()
            self._file.close()
