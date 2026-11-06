# Copyright (c) 2026-2027 zh
"""Task settings and the reference SuperMimic training configuration tree."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from .envs.velocity_env_cfg import EnvConfig
from .imagination.policy_trainer import PolicyTrainerCfg
from .policy.nets import CriticCfg, PolicyCfg
from .world_model.wm_trainer import WMTrainerCfg
from .world_model.world_model import WorldModelCfg

FRAMEWORK = "supermimic_velocity_v2"


@dataclass
class NoiseScales:
    """Uniform-noise half amplitudes in each observation term's units."""

    base_ang_vel: float = 0.2  # rad/s.
    projected_gravity: float = 0.05
    joint_pos: float = 0.01  # rad.
    joint_vel: float = 1.5  # rad/s.


@dataclass
class NoiseConfig:
    """Apply noise to each new proprioceptive frame once."""

    add_noise: bool = True
    noise_level: float = 1.0
    noise_scales: NoiseScales = field(default_factory=NoiseScales)


@dataclass
class CommandsConfig:
    """HIM sampling and heading feedback with body translation and world yaw tracking."""

    initial_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-2.0, 2.0),
    )
    resampling_time: tuple[float, float] = (4.0, 12.0)
    heading_range: tuple[float, float] = (-3.14, 3.14)
    heading_control_stiffness: float = 0.5
    rel_standing_envs: float = 0.1
    rel_forward_envs: float = 0.05


@dataclass
class GaitConfig:
    """Foot-name gait phases with soft contacts used only in imagination."""

    period: float = 0.6
    offsets: dict[str, float] = field(default_factory=lambda: {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0})
    stance_fraction: float = 0.56
    command_threshold: float = 0.1


@dataclass
class PostureConfig:
    walking_threshold: float = 0.1
    running_threshold: float = 1.5
    std_standing: dict[str, float] = field(default_factory=lambda: {"hip": 0.05, "thigh": 0.1, "calf": 0.15})
    std_walking: dict[str, float] = field(default_factory=lambda: {"hip": 0.15, "thigh": 0.35, "calf": 0.5})
    std_running: dict[str, float] = field(default_factory=lambda: {"hip": 0.15, "thigh": 0.35, "calf": 0.5})


@dataclass
class RewardScales:
    track_linear_velocity: float = 2.0
    track_angular_velocity: float = 2.0
    action_rate_l2: float = -0.1
    joint_pos_limits: float = -10.0
    body_ang_vel: float = -0.05
    foot_gait: float = 0.5
    feet_air_time: float = 0.2
    pose: float = 1.0
    lin_vel_z: float = -2.0
    body_orientation_l2: float = -0.2
    smoothness: float = -0.01
    stand_still: float = -2.0
    stand_still_joint_vel_l2: float = -1.0
    foot_slip: float = -0.1
    joint_acc_l2: float = -3.0e-7
    stumble: float = -0.1
    joint_torques_l2: float = -2.0e-5
    leg_collision: float = -1.0
    angular_momentum: float = -0.025
    hip_pos: float = -1.0
    soft_landing: float = -3.0e-3


@dataclass
class RewardConfig:
    """Body-frame translation and HIM reward rates."""

    tracking_linear_variance: float = 0.25
    tracking_angular_variance: float = 0.25
    fall_angle: float = math.radians(70)
    fall_softness_deg: float = 5.0
    stand_still_command_threshold: float = 0.1
    foot_slip_command_threshold: float = 0.1
    stumble_ratio: float = 5.0
    leg_collision_force_threshold: float = 1.0
    hip_pos_command_threshold: float = 0.1
    soft_landing_command_threshold: float = 0.1
    gait: GaitConfig = field(default_factory=GaitConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    scales: RewardScales = field(default_factory=RewardScales)


@dataclass
class TaskConfig:
    """Go2 environment, observations, commands, and rewards."""

    env: EnvConfig = field(default_factory=EnvConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    commands: CommandsConfig = field(default_factory=CommandsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, values: dict) -> TaskConfig:
        cfg = cls()
        _update_config(cfg, values)
        return cfg


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000
    updates_per_env_step: float = 0.4
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50
    checkpoint_interval: int = 4000
    max_hours: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "go2_velocity"
    log_root: Path = Path("logs/supermimic")
    device: str = "cuda:0"
    task: TaskConfig = field(default_factory=TaskConfig)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None

    @classmethod
    def from_dict(cls, values: dict) -> TrainConfig:
        cfg = cls()
        _update_config(cfg, values)
        return cfg


def _update_config(target, values: dict) -> None:
    """Read declared fields from this protocol's JSON configuration."""
    if not isinstance(values, dict):
        raise ValueError("Configuration sections must be objects")
    declared = {item.name for item in fields(target)}
    for name, value in values.items():
        if name not in declared:
            raise ValueError(f"Unknown configuration field: {name}")
        current = getattr(target, name)
        if is_dataclass(current):
            _update_config(current, value)
        elif isinstance(current, Path):
            setattr(target, name, Path(value))
        else:
            if isinstance(current, tuple):
                value = tuple(value)
            setattr(target, name, deepcopy(value))


def robot_settings() -> dict:
    from src.assets.robots.unitree_go2.go2_constants import (
        DELAYED_CFG,
        get_go2_robot_cfg,
    )

    settings = asdict(get_go2_robot_cfg(DELAYED_CFG))
    del settings["spec_fn"]
    return settings


def load_config(path: str | Path, *, snapshot: bool = False) -> tuple[TaskConfig, dict]:
    """Load task settings from a current-protocol snapshot."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Task configuration must be an object")
    metadata = {name: value for name, value in payload.items() if name != "config"}
    if snapshot and metadata.get("framework") != FRAMEWORK:
        raise ValueError("This checkpoint predates the reference training protocol; use its original checkout")
    if "robot" in metadata and metadata["robot"] != json.loads(json.dumps(robot_settings())):
        raise ValueError("Saved robot settings differ from the native Go2 preset")
    return TaskConfig.from_dict(payload.get("config", payload)), metadata
