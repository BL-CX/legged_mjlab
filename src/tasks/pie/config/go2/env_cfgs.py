"""Unitree Go2 rough-terrain PIE environment configuration."""

from __future__ import annotations

import copy
import math

from mjlab.envs import ManagerBasedRlEnvCfg, mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    CameraSensorCfg,
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.assets.robots import get_go2_robot_cfg
from src.tasks.pie import mdp
from src.tasks.pie.mdp.curriculums import TerrainVelocityCurriculum, terrain_levels_vel
from src.tasks.pie.mdp.velocity_command import TerrainVelocityCommandCfg
from src.tasks.pie.terrains import ROUGH_TERRAINS_CFG


_FOOT_NAMES = ("FR", "FL", "RR", "RL")
_FOOT_GEOMS = tuple(f"{name}_foot_collision" for name in _FOOT_NAMES)
_FOOT_RAYS = tuple(f"{name}_foot_ray" for name in _FOOT_NAMES)

_PIE_DEPTH_RAW_WIDTH = 106
_PIE_DEPTH_HEIGHT = 60
_PIE_DEPTH_CROP_LEFT = 10
_PIE_DEPTH_CROP_RIGHT = 10
_PIE_DEPTH_HORIZONTAL_FOV_DEG = 87.0


def _horizontal_to_vertical_fov(horizontal_fov: float) -> float:
    return 2.0 * math.degrees(
        math.atan(
            math.tan(math.radians(horizontal_fov) * 0.5)
            * _PIE_DEPTH_HEIGHT
            / _PIE_DEPTH_RAW_WIDTH
        )
    )


_PIE_DEPTH_FOVY_DEG = _horizontal_to_vertical_fov(_PIE_DEPTH_HORIZONTAL_FOV_DEG)
_PIE_DEPTH_FOVY_RANDOM_RANGE = (
    _horizontal_to_vertical_fov(_PIE_DEPTH_HORIZONTAL_FOV_DEG - 1.0)
    - _PIE_DEPTH_FOVY_DEG,
    _horizontal_to_vertical_fov(_PIE_DEPTH_HORIZONTAL_FOV_DEG + 1.0)
    - _PIE_DEPTH_FOVY_DEG,
)


def _proprio_terms(*, noisy: bool) -> dict[str, ObservationTermCfg]:
    """Build current body, command, joint and action observations."""
    return {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2) if noisy else None,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05) if noisy else None,
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            noise=Unoise(n_min=-0.01, n_max=0.01) if noisy else None,
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            noise=Unoise(n_min=-1.5, n_max=1.5) if noisy else None,
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }


def rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Go2 visual locomotion environment."""
    # Sensors.
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.7, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0, 1, 2),
        debug_vis=False,
    )

    foot_rays = tuple(
        RayCastSensorCfg(
            name=f"{foot}_foot_ray",
            frame=ObjRef(type="site", name=foot, entity="robot"),
            ray_alignment="world",
            pattern=GridPatternCfg(size=(0.02, 0.02), resolution=0.02),
            max_distance=0.6,
            exclude_parent_body=True,
            include_geom_groups=(0, 1, 2),
            debug_vis=False,
        )
        for foot in _FOOT_NAMES
    )

    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=_FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
        history_length=4,
    )
    nonfoot_ground = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern=r".*_(thigh|calf)$"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=4,
    )
    base_ground = ContactSensorCfg(
        name="base_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="base_link"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=4,
    )

    depth_camera = CameraSensorCfg(
        name="front_depth",
        parent_body="robot/base_link",
        pos=(0.345, 0.0, 0.07),
        quat=(0.5792280, 0.4055798, -0.4055798, -0.5792280),
        fovy=_PIE_DEPTH_FOVY_DEG,
        width=_PIE_DEPTH_RAW_WIDTH,
        height=_PIE_DEPTH_HEIGHT,
        data_types=("depth",),
        enabled_geom_groups=(0, 1, 2),
        use_shadows=False,
        use_textures=False,
    )

    # Observations.
    noisy_proprio = _proprio_terms(noisy=True)
    clean_proprio = _proprio_terms(noisy=False)
    observations = {
        "actor": ObservationGroupCfg(
            terms=noisy_proprio,
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=1,
        ),
        "proprio_history": ObservationGroupCfg(
            terms=copy.deepcopy(noisy_proprio),
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=10,
            flatten_history_dim=True,
        ),
        "camera": ObservationGroupCfg(
            terms={
                "front_depth": ObservationTermCfg(
                    func=mdp.DepthHistory,
                    params={
                        "sensor_name": depth_camera.name,
                        "cutoff_distance": 3.0,
                        "crop_left": _PIE_DEPTH_CROP_LEFT,
                        "crop_right": _PIE_DEPTH_CROP_RIGHT,
                        "gaussian_blur": (3, 1.0),
                        "frame_history_length": 2,
                        "update_period_steps": 5,
                    },
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
            history_length=None,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            terms={
                **copy.deepcopy(clean_proprio),
                "base_lin_vel": ObservationTermCfg(
                    func=mdp.builtin_sensor,
                    params={"sensor_name": "robot/imu_lin_vel"},
                ),
                "height_scan": ObservationTermCfg(
                    func=envs_mdp.height_scan,
                    params={"sensor_name": terrain_scan.name},
                    scale=1.0 / terrain_scan.max_distance,
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
        "velocity_target": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(
                    func=mdp.builtin_sensor,
                    params={"sensor_name": "robot/imu_lin_vel"},
                )
            },
            enable_corruption=False,
            history_length=1,
        ),
        "height_target": ObservationGroupCfg(
            terms={
                "height_scan": ObservationTermCfg(
                    func=envs_mdp.height_scan,
                    params={"sensor_name": terrain_scan.name},
                    scale=1.0 / terrain_scan.max_distance,
                )
            },
            enable_corruption=False,
            history_length=1,
        ),
        "foot_clearance_target": ObservationGroupCfg(
            terms={
                "foot_clearance": ObservationTermCfg(
                    func=mdp.foot_clearance,
                    params={"sensor_names": _FOOT_RAYS, "max_clearance": 0.6},
                    scale=1.0 / 0.6,
                )
            },
            enable_corruption=False,
            history_length=1,
        ),
        "successor_target": ObservationGroupCfg(
            terms=copy.deepcopy(clean_proprio),
            enable_corruption=False,
            history_length=1,
        ),
    }

    # Actions and commands.
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True,
        )
    }
    commands: dict[str, CommandTermCfg] = {
        "twist": TerrainVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=True,
            heading_control_stiffness=2.0,
            terrain_ranges={
                "smooth_slope": TerrainVelocityCommandCfg.Ranges(
                    lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-math.pi, math.pi), heading=(-math.pi, math.pi),
                    lin_vel_x_limits=(-1.5, 1.5), lin_vel_y_limits=(-1.0, 1.0),
                ),
                "smooth_slope_inv": TerrainVelocityCommandCfg.Ranges(
                    lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-math.pi, math.pi), heading=(-math.pi, math.pi),
                    lin_vel_x_limits=(-1.5, 1.5), lin_vel_y_limits=(-1.0, 1.0),
                ),
                "rough_flat": TerrainVelocityCommandCfg.Ranges(
                    lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-math.pi, math.pi), heading=(-math.pi, math.pi),
                    lin_vel_x_limits=(-2.0, 2.0), lin_vel_y_limits=(-1.0, 1.0),
                ),
                "discrete_obstacles": TerrainVelocityCommandCfg.Ranges(
                    lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-math.pi, math.pi), heading=(-math.pi, math.pi),
                    lin_vel_x_limits=(-1.0, 1.0), lin_vel_y_limits=(-0.8, 0.8),
                ),
                "pyramid_stairs_free": TerrainVelocityCommandCfg.Ranges(
                    lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-math.pi, math.pi), heading=(-math.pi, math.pi),
                    lin_vel_x_limits=(-1.0, 1.0), lin_vel_y_limits=(-0.8, 0.8),
                ),
                "pyramid_stairs_inv_free": TerrainVelocityCommandCfg.Ranges(
                    lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-math.pi, math.pi), heading=(-math.pi, math.pi),
                    lin_vel_x_limits=(-1.0, 1.0), lin_vel_y_limits=(-0.8, 0.8),
                ),
                "pyramid_stairs_directed": TerrainVelocityCommandCfg.Ranges(
                    lin_vel_x=(0.0, 0.5), lin_vel_y=(0.0, 0.0),
                    ang_vel_z=(-math.pi, math.pi), heading=(0.0, 0.0),
                    lin_vel_x_limits=(0.0, 1.0), lin_vel_y_limits=(0.0, 0.0),
                    zero_command_probability=0.0, min_lin_speed=0.2,
                ),
                "pyramid_stairs_inv_directed": TerrainVelocityCommandCfg.Ranges(
                    lin_vel_x=(0.0, 0.5), lin_vel_y=(0.0, 0.0),
                    ang_vel_z=(-math.pi, math.pi), heading=(0.0, 0.0),
                    lin_vel_x_limits=(0.0, 1.0), lin_vel_y_limits=(0.0, 0.0),
                    zero_command_probability=0.0, min_lin_speed=0.2,
                ),
            },
        ),
    }

    # Reset and domain randomization.
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.15, 0.15),
                "velocity_range": (-0.1, 0.1),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 8.0),
            params={
                "velocity_range": {
                    "x": (-0.4, 0.4),
                    "y": (-0.4, 0.4),
                    "z": (-0.25, 0.25),
                    "roll": (-0.3, 0.3),
                    "pitch": (-0.3, 0.3),
                    "yaw": (-0.4, 0.4),
                }
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.2, 1.2),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                "operation": "add",
                "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        ),
        "base_payload": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                "operation": "add",
                "ranges": (-1.0, 2.0),
            },
        ),
        "pd_gains": EventTermCfg(
            mode="startup",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.9, 1.1),
                "kd_range": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "motor_strength": EventTermCfg(
            mode="startup",
            func=dr.effort_limits,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "effort_limit_range": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "camera_position": EventTermCfg(
            mode="startup",
            func=dr.cam_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", camera_names=(depth_camera.name,)),
                "operation": "add",
                "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
            },
        ),
        "camera_pitch": EventTermCfg(
            mode="startup",
            func=dr.cam_quat,
            params={
                "asset_cfg": SceneEntityCfg("robot", camera_names=(depth_camera.name,)),
                "pitch_range": (-math.radians(1.0), math.radians(1.0)),
            },
        ),
        "camera_fovy": EventTermCfg(
            mode="startup",
            func=dr.cam_fovy,
            params={
                "asset_cfg": SceneEntityCfg("robot", camera_names=(depth_camera.name,)),
                "operation": "add",
                "ranges": _PIE_DEPTH_FOVY_RANDOM_RANGE,
            },
        ),
    }

    # Rewards, terminations and curriculum.
    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=1.0,
            params={"command_name": "twist", "std": 0.5},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=0.5,
            params={"command_name": "twist", "std": 0.5},
        ),
        "joint_torques_l2": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "smoothness": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01),
        "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0),
        "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
        "orientation": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.2),
        "collision": RewardTermCfg(
            func=mdp.collision_count,
            weight=-1.0,
            params={"sensor_name": nonfoot_ground.name, "force_threshold": 0.1},
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": math.radians(55.0)},
        ),
        "illegal_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": base_ground.name, "force_threshold": 10.0},
        ),
    }

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=terrain_levels_vel,
            params={"command_name": "twist"},
        ),
        "command_velocity": CurriculumTermCfg(
            func=TerrainVelocityCurriculum,
            params={
                "command_name": "twist",
                "reward_term_name": "track_linear_velocity",
                "threshold": 0.8,
                "increment": 0.2,
            },
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=copy.deepcopy(ROUGH_TERRAINS_CFG),
                max_init_terrain_level=0,
            ),
            entities={"robot": get_go2_robot_cfg()},
            sensors=(terrain_scan, *foot_rays, feet_ground, nonfoot_ground, base_ground, depth_camera),
            num_envs=1 if play else 256,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.8,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=48,
            njmax=1800,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=500,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.curriculum = {}
        for name in ("push_robot", "camera_position", "camera_pitch", "camera_fovy"):
            cfg.events.pop(name, None)
        # Preserve fixed terrain columns so commands match their terrain names.
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain, mode="reset", params={},
        )

    return cfg
