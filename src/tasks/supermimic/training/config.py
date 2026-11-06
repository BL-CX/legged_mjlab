"""Learning settings and native environment composition for SuperMimic training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..imagination.policy_trainer import PolicyTrainerCfg
from ..policy.nets import CriticCfg, PolicyCfg
from ..world_model.wm_trainer import WMTrainerCfg
from ..world_model.world_model import WorldModelCfg


@dataclass
class ScheduleCfg:
    total_iterations: int = 20_000
    updates_per_env_step: float = 0.4
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50
    checkpoint_interval: int = 4000
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


@dataclass
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: SuperMimicCfg
    run_name: str = "velocity"
    log_root: Path = Path("logs/supermimic")
    device: str = "cuda:0"
    resume: Path | None = None
