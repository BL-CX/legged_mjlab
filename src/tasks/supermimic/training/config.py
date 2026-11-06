"""Compose native environment settings with the quadruped learning protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..imagination.policy_trainer import PolicyTrainerCfg
from ..policy.nets import CriticCfg, PolicyCfg
from ..world_model.wm_trainer import WMTrainerCfg
from ..world_model.world_model import WorldModelCfg

CHECKPOINT_FORMAT = "supermimic_quadruped_v3"


@dataclass
class ScheduleCfg:
    total_iterations: int = 40_000
    collect_steps: float = 2.5
    warmup_iterations: int = 300
    actor_ramp_iterations: int = 1000
    lr_decay_iterations: int = 30000
    lr_final_ratio: float = 0.03
    collapse_falls: float = 3.0
    collapse_interval: int = 50
    log_interval: int = 50
    checkpoint_interval: int = 1000
    max_hours: float | None = None


@dataclass
class SuperMimicCfg:
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    action_noise_std: float = 0.1
    explore_fraction: float = 0.25
    explore_noise: float = 0.3
    explore_theta: float = 2.0
    control_enabled: bool = True


@dataclass
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: SuperMimicCfg
    run_name: str = "velocity"
    log_root: Path = Path("logs/supermimic")
    device: str = "cuda:0"
    resume: Path | None = None
