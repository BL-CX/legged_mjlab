"""Go2 velocity environment, observations, commands, and rewards."""

from __future__ import annotations

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots import get_go2_robot_cfg

from ...data.recorder import ReplayRecorderCfg
from ...envs.mdp.commands import (
    VelocityCommandCfg,
    initialize_terrain,
    observation,
    reset_origins,
    reward_term,
    task_termination,
    task_time_out,
)
from ...envs.mdp.terrains import clean_look, reference_generator


def rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the native Go2 velocity environment with reference terrain geometry."""
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
            primary=ContactMatch(
                mode="geom",
                pattern=r".*_(hip|thigh|calf).*_collision\d*$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=1,
        ),
    )
    scene = SceneCfg(
        entities={"robot": get_go2_robot_cfg()},
        terrain=clean_look(
            TerrainEntityCfg(
                terrain_type="generator",
                spec_fn=mujoco.MjSpec,
                terrain_generator=reference_generator(),
                max_init_terrain_level=None,
            )
        ),
        sensors=sensors,
        num_envs=4096,
        extent=2.0,
    )

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            use_default_offset=True,
            scale={".*_hip_joint": 0.25, ".*_thigh_joint": 0.5, ".*_calf_joint": 0.5},
        ),
    }

    events = {
        "terrain_origins": EventTermCfg(func=reset_origins, mode="reset"),
        "reset_root": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.01, 0.05),
                    "yaw": (-3.14, 3.14),
                },
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
                    "ranges": {
                        0: (-0.025, 0.025),
                        1: (-0.025, 0.025),
                        2: (-0.03, 0.03),
                    },
                },
            ),
        }
    )

    events["task_startup"] = EventTermCfg(func=initialize_terrain, mode="startup")

    terminations = {"task": TerminationTermCfg(func=task_termination)}
    terminations["time_out"] = TerminationTermCfg(func=task_time_out, time_out=True)
    observations = {
        "observation": ObservationGroupCfg(
            terms={"frame": ObservationTermCfg(func=observation)},
            enable_corruption=False,
        ),
        "history": ObservationGroupCfg(
            terms={"frame": ObservationTermCfg(func=observation)},
            history_length=6,
            flatten_history_dim=False,
            enable_corruption=False,
        ),
    }
    rewards = {
        name: RewardTermCfg(func=reward_term, weight=weight, params={"name": name})
        for name, weight in {
            "track_linear_velocity": 2.0,
            "track_angular_velocity": 2.0,
            "action_rate_l2": -0.1,
            "joint_pos_limits": -10.0,
            "body_ang_vel": -0.05,
            "foot_gait": 0.5,
            "feet_air_time": 0.2,
            "pose": 1.0,
            "lin_vel_z": -2.0,
            "body_orientation_l2": -0.2,
            "smoothness": -0.01,
            "stand_still": -2.0,
            "stand_still_joint_vel_l2": -1.0,
            "foot_slip": -0.1,
            "joint_acc_l2": -3.0e-7,
            "stumble": -0.1,
            "joint_torques_l2": -2.0e-5,
            "leg_collision": -1.0,
            "angular_momentum": -0.025,
            "hip_pos": -1.0,
            "soft_landing": -3.0e-3,
        }.items()
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        events=events,
        commands={"velocity": VelocityCommandCfg()},
        rewards=rewards,
        recorders={"replay": ReplayRecorderCfg()},
        curriculum={},
        metrics={},
        terminations=terminations,
        auto_reset=True,
        decimation=4,
        episode_length_s=20.0,
        seed=0,
        sim=SimulationCfg(
            nconmax=None,
            njmax=300,
            broadphase="nxn",
            mujoco=MujocoCfg(
                timestep=0.005,
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

    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        command = cfg.commands["velocity"]
        command.play = True
        command.noise.add_noise = False
        command.terrain_curriculum = False
        cfg.terminations.pop("time_out")
        cfg.recorders = {}
    return cfg


def flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Go2 velocity control on a plane with the same task mathematics."""
    cfg = rough_env_cfg(play=play)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    return cfg
