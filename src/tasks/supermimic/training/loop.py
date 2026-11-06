# Copyright (c) 2026-2027 zh
"""SuperMimic training: collect one environment step, fit dynamics, improve policy."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.utils.torch import configure_torch_backends
from torch.utils.tensorboard import SummaryWriter

from ..config import FRAMEWORK, robot_settings
from ..config.go2.env_cfg import make_env
from ..config.go2.rl_cfg import TrainConfig
from ..imagination.policy_trainer import PolicyTrainer
from ..policy.nets import Critic, Policy
from ..world_model.wm_trainer import WorldModelTrainer
from ..world_model.world_model import WorldModel
from .distributed import allreduce_mean, broadcast_params, finalize, init_distributed
from .schedule import cosine_lr


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg = deepcopy(cfg)
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            if cfg.env.terrain == "rough" and cfg.env.terrain_seed is None:
                cfg.env.terrain_seed = cfg.env.seed
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        configure_torch_backends()
        env_cfg = deepcopy(cfg.env)
        env_cfg.seed += 1000 * self.rank
        torch.manual_seed(env_cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_dir = Path(cfg.log_root).expanduser().resolve() / cfg.run_name
        self.writer = None
        self.env = None
        try:
            self.env = env = make_env(env_cfg, device=str(self.device), record=True)
            self.command = env.command_manager.get_term("velocity")
            self.task = self.command.model
            self.wm = WorldModel(cfg.world_model, self.task, self.device)
            self.policy = Policy(cfg.policy, history_length=cfg.env.history_length).to(self.device)
            self.critic = Critic(cfg.critic).to(self.device)
            for module in (self.wm, self.policy, self.critic):
                broadcast_params(module)
            self.wm_trainer = WorldModelTrainer(self.wm, cfg.wm_trainer)
            self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, cfg.policy_trainer)
            self.recorder = env.recorder_manager.get_term("replay")
            self.buffer = self.recorder.buffer
            self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
            self.ep_len = torch.zeros(env.num_envs, device=self.device)
            self.iteration = self.start_iteration = self.num_updates = 0
            self._utd_acc = 0.0
            if cfg.resume is not None:
                self.load(cfg.resume)
            if self.is_main:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(str(self.log_dir / "tb"))
                (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
                snapshot = {
                    "framework": FRAMEWORK,
                    "config": asdict(self.command.env_cfg),
                    "spec_hash": self.task.hash,
                    "geometry_hash": self.command.surface.geometry_hash,
                    "robot": robot_settings(self.command.env_cfg),
                }
                (self.log_dir / "task_config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
                print(f"[INFO] Logging experiment in directory: {self.log_dir}", flush=True)
        except BaseException:
            self.close()
            raise

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        data = {
            "format": FRAMEWORK,
            "iteration": self.iteration,
            "num_updates": self.num_updates,
            "wm": self.wm.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "wm_opt": self.wm_trainer.opt.state_dict(),
            "policy_opt": self.policy_trainer.opt.state_dict(),
            "critic_opt": self.policy_trainer.critic_opt.state_dict(),
            "config": asdict(self.cfg),
            "spec_hash": self.task.hash,
            "curriculum": self.command.curriculum_state_dict(),
        }
        path = self.log_dir / name
        pending = path.with_suffix(path.suffix + ".tmp")
        torch.save(data, pending)
        os.replace(pending, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(Path(path).expanduser(), map_location=self.device, weights_only=False)
        if checkpoint.get("format") != FRAMEWORK:
            raise ValueError("Checkpoint uses an older training protocol; use its original checkout")
        saved = TrainConfig.from_dict(checkpoint["config"])
        for name in ("world_model", "wm_trainer", "policy", "critic", "policy_trainer"):
            if asdict(getattr(saved, name)) != asdict(getattr(self.cfg, name)):
                raise ValueError(f"Resume requires unchanged {name} settings")
        if checkpoint.get("spec_hash") != self.task.hash:
            raise ValueError("Resume requires unchanged task state, rewards, and native physics")
        self.wm.load_state_dict(checkpoint["wm"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.wm_trainer.opt.load_state_dict(checkpoint["wm_opt"])
        self.policy_trainer.opt.load_state_dict(checkpoint["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(checkpoint["critic_opt"])
        self.command.load_curriculum_state_dict(checkpoint["curriculum"])
        self.iteration = checkpoint["iteration"]
        self.start_iteration = self.iteration
        self.num_updates = checkpoint["num_updates"]
        print(f"[INFO] resumed from {path} at iteration {self.iteration}", flush=True)

    @torch.no_grad()
    def act(self, history):
        return self.policy(history)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run(self):
        cfg, sch, env = self.cfg, self.cfg.schedule, self.env
        try:
            if self.iteration >= sch.total_iterations:
                raise ValueError("Target iteration must exceed the checkpoint iteration")
            observations, _ = env.reset()
            history = observations["history"]
            t_start = time.time()
            timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
            agg, agg_n = {}, 0
            finished_returns, finished_lens = [], []
            while self.iteration < sch.total_iterations:
                t0 = time.perf_counter()
                action = self.act(history)
                executed = action + cfg.env.action_noise_std * torch.randn_like(action)
                self.recorder.clean_action = action
                observations, reward, terminated, truncated, _ = env.step(executed)
                done = terminated | truncated
                self.ep_reward_sum += reward
                self.ep_len += 1
                if done.any():
                    finished_returns.extend(self.ep_reward_sum[done].tolist())
                    finished_lens.extend(self.ep_len[done].tolist())
                    self.ep_reward_sum[done] = 0
                    self.ep_len[done] = 0
                history = observations["history"]
                self._synchronize()
                timers["sim"] += time.perf_counter() - t0
                self.iteration += 1

                if self.iteration - self.start_iteration >= sch.warmup_iterations:
                    self._utd_acc += sch.updates_per_env_step
                    while self._utd_acc >= 1.0:
                        self._utd_acc -= 1.0
                        lr = cosine_lr(
                            self.iteration,
                            sch.total_iterations,
                            cfg.wm_trainer.lr,
                            sch.lr_final_ratio,
                            sch.lr_decay_fraction,
                        )
                        self.wm_trainer.set_lr(lr)
                        self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                        t0 = time.perf_counter()
                        m_wm = self.wm_trainer.update(self.buffer)
                        self._synchronize()
                        timers["wm"] += time.perf_counter() - t0
                        t0 = time.perf_counter()
                        for _ in range(sch.policy_updates_per_wm_update):
                            m_pi = self.policy_trainer.update(self.buffer)
                        self._synchronize()
                        timers["policy"] += time.perf_counter() - t0
                        self.num_updates += 1
                        for key, value in m_wm.items():
                            agg[f"wm/{key}"] = agg.get(f"wm/{key}", 0.0) + value
                        for key, value in m_pi.items():
                            agg[f"policy/{key}"] = agg.get(f"policy/{key}", 0.0) + value
                        agg["lr"] = lr
                        agg_n += 1

                if self.iteration % sch.log_interval == 0:
                    if self.world > 1:
                        total = allreduce_mean(float(sum(finished_returns))) * self.world
                        count = allreduce_mean(float(len(finished_returns))) * self.world
                        finished_returns = [total / count] if count else []
                    if self.is_main:
                        for key, value in agg.items():
                            self.writer.add_scalar(key, value / (agg_n if key != "lr" else 1), self.iteration)
                        if finished_returns:
                            self.writer.add_scalar(
                                "train/episode_return", sum(finished_returns) / len(finished_returns), self.iteration
                            )
                            self.writer.add_scalar(
                                "train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), self.iteration
                            )
                        for key, value in timers.items():
                            self.writer.add_scalar(f"time/{key}_s_per_iter", value / sch.log_interval, self.iteration)
                        for key, value in self.command.last_metrics.items():
                            self.writer.add_scalar(key, value, self.iteration)
                        print(
                            f"it {self.iteration} | updates {self.num_updates} | replay {self.buffer.size} | "
                            f"wm {agg.get('wm/loss', 0) / max(agg_n, 1):.4f} | "
                            f"policy {agg.get('policy/imag_objective', 0) / max(agg_n, 1):.3f}",
                            flush=True,
                        )
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {key: 0.0 for key in timers}
                if self.iteration % sch.checkpoint_interval == 0:
                    self.save(f"ckpt_{self.iteration}.pt")
                if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                    stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                    if self.world > 1:
                        stop = allreduce_mean(stop)
                    if stop > 0:
                        break
            self.save()
            return self.log_dir
        finally:
            self.close()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.env is not None:
            self.env.close()
            self.env = None
        finalize()
