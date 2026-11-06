"""Collect real transitions, fit dynamics, and improve the quadruped policy."""

from __future__ import annotations

import os
import time
from copy import deepcopy
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends

from ..imagination.policy_trainer import PolicyTrainer
from ..policy.nets import Critic, Policy
from ..world_model.wm_trainer import WorldModelTrainer
from ..world_model.world_model import WorldModel
from .config import CHECKPOINT_FORMAT, TrainConfig
from .logger import TrainingLogger
from .schedule import cosine_lr


class Trainer:
    def __init__(self, cfg: TrainConfig):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError(
                "The aligned quadruped_lab protocol supports one training process; multi-rank normalization is not defined"
            )
        self.cfg = cfg = deepcopy(cfg)
        agent, schedule = cfg.agent, cfg.agent.schedule
        if schedule.collect_steps < 1 or cfg.env.recorders["replay"].params["capacity"] <= agent.wm_trainer.window + 1:
            raise ValueError("Collection steps or replay capacity are invalid")
        configure_torch_backends()
        env_cfg = deepcopy(cfg.env)
        torch.manual_seed(env_cfg.seed)
        self.device = torch.device(cfg.device)
        self.logger = self.env = None
        try:
            self.env = env = ManagerBasedRlEnv(env_cfg, device=str(self.device))
            self.command = env.command_manager.get_term("velocity")
            self.task = self.command.model
            self.policy = Policy(agent.policy, history_length=env.cfg.observations["history"].history_length).to(
                self.device
            )
            self.wm = WorldModel(agent.world_model, self.task, self.device)
            self.critic = Critic(agent.critic).to(self.device)
            self.wm_trainer = WorldModelTrainer(self.wm, agent.wm_trainer)
            self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, agent.policy_trainer)
            self.recorder = env.recorder_manager.get_term("replay")
            self.buffer = self.recorder.buffer
            self.ou_noise = torch.zeros(env.num_envs, self.task.spec.act_dim, device=self.device)
            self.iteration = self.num_updates = self.real_steps = 0
            self._collect_accumulator = 0.0
            self.policy_start = schedule.warmup_iterations
            self.actor_frozen_until = self.ramp_start = 0
            self.lr_multiplier = 1.0
            self._guard_falls = 0.0
            self._guard_reset_steps = self._collapse_strikes = 0
            self._safe_policy = None
            self.logger = TrainingLogger(cfg, num_envs=env.num_envs, device=self.device, step_dt=env.step_dt)
            self.log_dir = self.logger.log_dir
            self.logger.message(f"[INFO] Logging experiment in directory: {self.log_dir}")
            if cfg.resume is not None:
                self.load(cfg.resume)
        except BaseException:
            self.close()
            raise

    def save(self, name="ckpt_latest.pt"):
        """Publish atomically only while the collapse guard permits it."""
        if self._collapse_strikes:
            self.logger.message(f"[WARN] Skipped {name}: collapse guard has not cleared")
            return
        data = {
            "format": CHECKPOINT_FORMAT,
            "iteration": self.iteration,
            "num_updates": self.num_updates,
            "real_steps": self.real_steps,
            "wm": self.wm.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "wm_opt": self.wm_trainer.opt.state_dict(),
            "policy_opt": self.policy_trainer.opt.state_dict(),
            "critic_opt": self.policy_trainer.critic_opt.state_dict(),
            "config": self.cfg,
            "curriculum": self.command.curriculum_state_dict(),
        }
        path = self.log_dir / name
        pending = path.with_suffix(path.suffix + ".tmp")
        torch.save(data, pending)
        os.replace(pending, path)
        self._safe_policy = {
            name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in (("policy", self.policy), ("critic", self.critic))
        }

    def load(self, path):
        checkpoint = torch.load(Path(path).expanduser(), map_location=self.device, weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Unsupported checkpoint format; start a new run with the current configuration")
        for name, module in (("wm", self.wm), ("policy", self.policy), ("critic", self.critic)):
            module.load_state_dict(checkpoint[name])
        self.wm_trainer.opt.load_state_dict(checkpoint["wm_opt"])
        self.policy_trainer.opt.load_state_dict(checkpoint["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(checkpoint["critic_opt"])
        self.iteration = checkpoint["iteration"]
        self.num_updates = checkpoint["num_updates"]
        self.real_steps = checkpoint["real_steps"]
        # Resume progress; physics, replay, and terrain assignments restart.
        self.command.command_steps = self.iteration
        self.env.common_step_counter = self.iteration
        self.logger.message(f"[INFO] Resumed from {path} at iteration {self.iteration}")

    @torch.no_grad()
    def act(self, history):
        return self.policy(history)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def collect(self, history, steps):
        """Record actual noisy histories before updating real-transition statistics."""
        cfg, env = self.cfg.agent, self.env
        metrics, counts = {}, {}
        failures = timeouts = reset_steps = 0
        for _ in range(steps):
            before = self.task.spec.pack(self.command.state).clone()
            xi = self.command.state.context.clone()
            action = self.act(history)
            self.ou_noise.mul_(1 - cfg.explore_theta * env.step_dt).add_(
                cfg.explore_noise * (2 * cfg.explore_theta) ** 0.5 * env.step_dt**0.5 * torch.randn_like(action)
            )
            noise = cfg.action_noise_std * torch.randn_like(action)
            exploration_count = int(env.num_envs * cfg.explore_fraction)
            noise[:exploration_count] = self.ou_noise[:exploration_count]
            executed = action + noise
            if not torch.isfinite(executed).all():
                raise FloatingPointError("Collected actions are non-finite")
            self.recorder.clean_action = action
            observations, reward, terminated, truncated, extras = env.step(executed)
            self.policy.update_statistics(history)
            done = terminated | truncated
            keep = ~done
            if keep.any():
                final = self.task.spec.pack(self.command.final_state)
                self.wm.update_stats(before[keep], executed[keep], final[keep], xi[keep])
            self.logger.process_env_step(reward, done)
            if done.any():
                reset_steps += 1
            failures += int(terminated.sum())
            timeouts += int((truncated & ~terminated).sum())
            self.ou_noise[done] = 0
            history = observations["history"]
            self.real_steps += env.num_envs
            # Native episode summaries occur only on reset; average each key over its own samples.
            for key, value in (extras.get("log", {}) | self.command.last_metrics).items():
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    value = value.float().mean()
                metrics[key] = metrics.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        metrics = {key: value / counts[key] for key, value in metrics.items()}
        metrics.update(self.logger.episode_statistics())
        metrics.update(failures=float(failures), timeouts=float(timeouts), reset_steps=float(reset_steps))
        return history, metrics

    def _collapse_guard(self, metrics):
        """Restore saved policy and critic after two high-failure intervals."""
        cfg = self.cfg.agent.schedule
        self._guard_falls += metrics["failures"]
        self._guard_reset_steps += metrics["reset_steps"]
        if self.iteration % cfg.collapse_interval:
            return
        falls = self._guard_falls / max(self._guard_reset_steps, 1)
        self._guard_falls, self._guard_reset_steps = 0.0, 0
        metrics["guard/falls_per_reset_step"] = falls
        if self.iteration <= self.actor_frozen_until or falls <= cfg.collapse_falls:
            self._collapse_strikes = 0
            return
        self._collapse_strikes += 1
        if self._collapse_strikes < 2 or self._safe_policy is None:
            return
        self.policy.load_state_dict(self._safe_policy["policy"])
        self.critic.load_state_dict(self._safe_policy["critic"])
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.cfg.agent.policy_trainer)
        self.lr_multiplier *= 0.5
        self.ramp_start = self.iteration
        self._collapse_strikes = 0
        metrics["guard/rollback"] = 1.0
        self.logger.message(f"[WARN] Collapse guard restored actor and critic; lr scale {self.lr_multiplier:.3g}")

    def run(self):
        cfg, sch, env = self.cfg.agent, self.cfg.agent.schedule, self.env
        try:
            if self.iteration >= sch.total_iterations:
                raise ValueError("Target iteration must exceed the checkpoint iteration")
            observations, _ = env.reset()
            history = observations["history"]
            self.logger.message("[INFO] Prefilling replay before world model training")
            while not self.buffer.ready(cfg.wm_trainer.window):
                history, prefill = self.collect(history, int(sch.collect_steps))
                self._guard_falls += prefill["failures"]
                self._guard_reset_steps += prefill["reset_steps"]
            self._synchronize()
            started = time.perf_counter()
            self.logger.begin(self.iteration, self.real_steps)
            while self.iteration < sch.total_iterations:
                self.iteration += 1
                t0 = time.perf_counter()
                self._collect_accumulator += sch.collect_steps
                steps = int(self._collect_accumulator)
                self._collect_accumulator -= steps
                history, metrics = self.collect(history, steps)
                self._synchronize()
                metrics["time/collection_seconds"] = time.perf_counter() - t0
                lr = (
                    cosine_lr(self.iteration, sch.lr_decay_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio)
                    * self.lr_multiplier
                )
                ramp_start = max(self.ramp_start, self.actor_frozen_until, self.policy_start)
                ramp = min(1.0, max(0, self.iteration - ramp_start) / max(1, sch.actor_ramp_iterations))
                self.wm_trainer.set_lr(lr)
                self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr, ramp)
                t0 = time.perf_counter()
                metrics.update({f"wm/{key}": value for key, value in self.wm_trainer.update(self.buffer).items()})
                self._synchronize()
                metrics["time/wm_seconds"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                if cfg.control_enabled and self.iteration > self.policy_start:
                    metrics.update(
                        {
                            f"policy/{key}": value
                            for key, value in self.policy_trainer.update(
                                self.buffer, freeze_actor=self.iteration <= self.actor_frozen_until
                            ).items()
                        }
                    )
                self._synchronize()
                metrics["time/policy_seconds"] = time.perf_counter() - t0
                self.num_updates += 1
                metrics.update(
                    lr=lr,
                    iteration=self.iteration,
                    real_steps=self.real_steps,
                    replay_rows=self.buffer.size * env.num_envs,
                    archive_size=self.buffer.archive.size,
                    actor_lr=self.policy_trainer.opt.param_groups[0]["lr"],
                    critic_lr=self.policy_trainer.critic_opt.param_groups[0]["lr"],
                    actor_ramp=ramp,
                )
                self._collapse_guard(metrics)
                if not cfg.control_enabled:
                    phase = "world-model only"
                elif self.iteration <= self.policy_start:
                    phase = f"world-model warmup (policy starts at {self.policy_start + 1})"
                elif self.iteration <= self.actor_frozen_until:
                    phase = "critic training (actor frozen)"
                else:
                    phase = "actor ramp" if ramp < 1 else "joint training"
                self.logger.record(metrics, phase=phase)
                if self.iteration % sch.checkpoint_interval == 0:
                    self.save(f"ckpt_{self.iteration}.pt")
                if sch.max_hours is not None and time.perf_counter() - started >= sch.max_hours * 3600:
                    break
            elapsed = time.perf_counter() - started
            self.save()
            self.logger.message(
                f"[INFO] Completed {self.iteration} iterations; learning duration {elapsed:.3f} seconds"
            )
            return self.log_dir
        finally:
            self.close()

    def close(self):
        if self.logger is not None:
            self.logger.close()
            self.logger = None
        if self.env is not None:
            self.env.close()
            self.env = None
