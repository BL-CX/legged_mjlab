# Copyright (c) 2026-2027 zh
"""Native manager configuration and construction for quadruped velocity control."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots.unitree_go2.go2_constants import (
    ACTION_SCALE,
    DELAYED_CFG,
    get_go2_robot_cfg,
)

from .mdp.commands import (
    VelocityCommandCfg,
    initialize_terrain,
    observation,
    reset_origins,
    reward_term,
    task_termination,
)
from .mdp.terrains import TerrainMap

if TYPE_CHECKING:
    from ..config import TaskConfig


@dataclass
class EnvConfig:
    """Arguments for native physics construction and task terrain geometry."""

    num_envs: int = 4096
    scene_extent: float = 2.0
    seed: int = 0
    episode_length_s: float = 20.0
    action_noise_std: float = 0.1
    replay_capacity: int = 1000
    num_obs_history: int = 5
    timestep: float = 0.005
    decimation: int = 4
    action_scale: dict[str, float] = field(default_factory=lambda: ACTION_SCALE.copy())
    domain_randomization: bool = True
    terrain: str = "flat"
    terrain_seed: int | None = None
    terrain_curriculum: bool = True
    patch_size_m: tuple[float, float] = (8.0, 8.0)
    core_rows: int = 6
    num_cols: int = 20
    outer_flat_width_m: float = 20.0
    field_resolution_m: float = 0.02
    field_half_m: float = 6.5
    field_cache_dir: str = "logs/heightfields"


def make_env_cfg(
    cfg: TaskConfig | None = None,
    *,
    play: bool = False,
    no_terminations: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Configure native managers around the shared analytic task mathematics."""
    from ..config import TaskConfig

    cfg = deepcopy(cfg) if cfg is not None else TaskConfig()
    if play:
        cfg.noise.add_noise = False
        cfg.env.terrain_curriculum = False
    env = cfg.env
    terrain_map = TerrainMap(env)
    feet = r".*_foot_collision"
    sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=feet, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="leg_ground_contact",
            primary=ContactMatch(mode="geom", pattern=r".*_(hip|thigh|calf).*_collision\d*$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=1,
        ),
    )
    scene = SceneCfg(
        entities={"robot": get_go2_robot_cfg(DELAYED_CFG)},
        terrain=terrain_map.entity_cfg(),
        sensors=sensors,
        num_envs=env.num_envs,
        extent=env.scene_extent,
    )

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            use_default_offset=True,
            scale=env.action_scale,
        ),
    }

    events = {
        "terrain_origins": EventTermCfg(func=reset_origins, mode="reset"),
        "reset_root": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05), "yaw": (-3.14, 3.14)},
                "velocity_range": {},
            },
        ),
        "reset_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        ),
    }
    if env.domain_randomization:
        for name, axis, distribution, bounds in (
            ("slide", 0, "uniform", (0.3, 1.5)),
            ("spin", 1, "log_uniform", (1e-4, 2e-2)),
            ("roll", 2, "log_uniform", (1e-5, 5e-3)),
        ):
            events[f"foot_friction_{name}"] = EventTermCfg(
                func=envs_mdp.dr.geom_friction,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=feet),
                    "operation": "abs",
                    "distribution": distribution,
                    "ranges": tuple(bounds),
                    "axes": [axis],
                    "shared_random": True,
                },
            )
        events.update(
            {
                "encoder_bias": EventTermCfg(
                    func=envs_mdp.dr.encoder_bias,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot"),
                        "bias_range": (-0.015, 0.015),
                    },
                ),
                "base_com": EventTermCfg(
                    func=envs_mdp.dr.body_com_offset,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                        "operation": "add",
                        "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
                    },
                ),
            }
        )

    events["task_startup"] = EventTermCfg(func=initialize_terrain, mode="startup")

    terminations = {"task": TerminationTermCfg(func=task_termination)}
    if not play and not no_terminations:
        terminations["time_out"] = TerminationTermCfg(func=envs_mdp.time_out, time_out=True)
    observations = {
        "observation": ObservationGroupCfg(
            terms={"frame": ObservationTermCfg(func=observation)},
            enable_corruption=False,
        ),
        "history": ObservationGroupCfg(
            terms={"frame": ObservationTermCfg(func=observation)},
            history_length=env.num_obs_history,
            flatten_history_dim=False,
            enable_corruption=False,
        ),
    }
    rewards = {
        name: RewardTermCfg(func=reward_term, weight=weight, params={"name": name})
        for name, weight in vars(cfg.reward.scales).items()
    }

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        events=events,
        commands={
            "velocity": VelocityCommandCfg(
                task_cfg=cfg, terrain_map=terrain_map, play=play, no_terminations=no_terminations
            )
        },
        rewards=rewards,
        curriculum={},
        metrics={},
        terminations=terminations,
        auto_reset=True,
        decimation=env.decimation,
        episode_length_s=env.episode_length_s,
        seed=env.seed,
        sim=SimulationCfg(
            nconmax=None,
            njmax=300,
            broadphase="nxn",
            mujoco=MujocoCfg(
                timestep=env.timestep,
                solver="cg",
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
                impratio=1.0,
                cone="pyramidal",
            ),
        ),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.5,
            elevation=-10.0,
            azimuth=90.0,
        ),
    )


def make_env(
    cfg: TaskConfig | None = None,
    *,
    device: str = "cpu",
    play: bool = False,
    render_mode: str | None = None,
    video_height: int | None = None,
    video_width: int | None = None,
    no_terminations: bool = False,
    record: bool = False,
) -> ManagerBasedRlEnv:
    """Return the native environment; managers own all step and reset callbacks."""
    env_cfg = make_env_cfg(cfg, play=play, no_terminations=no_terminations)
    if video_height is not None:
        env_cfg.viewer.height = video_height
    if video_width is not None:
        env_cfg.viewer.width = video_width
    if record:
        from ..data.recorder import ReplayRecorderCfg

        env_cfg.recorders["replay"] = ReplayRecorderCfg(
            params={"capacity": env_cfg.commands["velocity"].task_cfg.env.replay_capacity}
        )
    return ManagerBasedRlEnv(env_cfg, device=device, render_mode=render_mode)
