"""Standalone Unitree Go2 HIM environment configurations."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
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
from src.envs import ManagerBasedRlEnvCfg
from src.tasks.loco import mdp
from src.tasks.loco.mdp import UniformVelocityCommandCfg
from src.tasks.loco.terrains import ROUGH_TERRAINS_CFG


_FOOT_NAMES = ("FR", "FL", "RR", "RL")
_FOOT_GEOMS = tuple(f"{name}_foot_collision" for name in _FOOT_NAMES)


def rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the self-contained Go2 HIM rough-terrain configuration."""

    ##
    # Sensors
    ##

    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        debug_vis=True,
        viz=RayCastSensorCfg.VizCfg(show_normals=True),
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=_FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    base_ground_cfg = ContactSensorCfg(
        name="base_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            # Only the three base collision geoms are termination contacts.
            pattern=r"base\d*_collision$",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    ##
    # Observations
    ##

    actor_terms = {
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
            scale=(2.0, 2.0, 0.25),
            clip=(-100.0, 100.0),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            scale=0.25,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            scale=1.0,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            scale=1.0,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
            scale=1.0,
            clip=(-100.0, 100.0),
        ),
    }
    clean_proprio = copy.deepcopy(actor_terms)
    for term in clean_proprio.values():
        term.noise = None

    critic_terms = {
        **copy.deepcopy(clean_proprio),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.base_lin_vel,
            scale=2.0,
            clip=(-100.0, 100.0),
        ),
        "base_external_force": ObservationTermCfg(
            func=mdp.base_external_force,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link"))},
            clip=(-100.0, 100.0),
        ),
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={
                "sensor_name": "terrain_scan", "offset": 0.5},
            clip=(-1.0, 1.0),
            scale=5.0,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=6,
            # Keep a time axis so concatenation produces [N, history, frame].
            flatten_history_dim=False,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=0,
        ),

    }

    ##
    # Metrics
    ##

    metrics = {
        "mean_action_acc": MetricsTermCfg(
            func=mdp.mean_action_acc,
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True,
        )
    }

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.05,
            rel_high_vel_envs=0.2,
            curriculums_limit_ranges=(-1.0, 2.0),
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )
    }

    ##
    # Events
    ##

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 6.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        "external_force": EventTermCfg(
            func=envs_mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(0.04, 0.04),
            params={
                "force_range": (-30.0, 30.0),
                "torque_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("base_link",),
                ),
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "operation": "abs",
                "ranges": (0.2, 1.6),
                "shared_random": True,  # All foot geoms share the same friction.
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
        "base_payload": EventTermCfg(
            func=dr.body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot",body_names=("base_link")),
                "operation": "add",
                "ranges": (-1.0, 2.0),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                "operation": "add",
                "ranges": {
                    0: (-0.05, 0.05),
                    1: (-0.05, 0.05),
                    2: (-0.05, 0.05),
                },
            },
        ),
    }

    ##
    # Rewards
    ##

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=1.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=0.5,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),

        "joint_power": RewardTermCfg(func=mdp.joint_power, weight=-2.0e-5),
        "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "smoothness": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01),
        "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
        "orientation": RewardTermCfg(
            func=mdp.body_orientation_l2,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
        ),
        "base_height": RewardTermCfg(
            func=mdp.base_height_reward,
            weight=-1.0,
            params={
                "target_height": 0.3,
                "sensor_name": "terrain_scan",
                "base_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            },
        ),

        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            weight=-0.01,
            params={
                "target_height": -0.20,
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_NAMES),
            },
        ),
        # "foot_slip": RewardTermCfg(
        #     func=mdp.feet_slip,
        #     weight=-0.25,
        #     params={
        #         "sensor_name": "feet_ground_contact",
        #         "command_name": "twist",
        #         "command_threshold": 0.1,
        #         "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_NAMES),
        #     },
        # ),
        # "stand_still": RewardTermCfg(
        #     func=mdp.stand_still,
        #     weight=-1.0,
        #     params={
        #         "command_name": "twist",
        #         "command_threshold": 0.1,
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        #     },
        # ),
    }

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        # "fell_over": TerminationTermCfg(
        #     func=mdp.bad_orientation,
        #     params={"limit_angle": math.radians(70.0)},
        # ),
        "illegal_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": base_ground_cfg.name, "force_threshold": 10.0},
        ),
    }

    ##
    # Curriculum
    ##

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        ),
        "command_vel": CurriculumTermCfg(
            func=mdp.lin_vel_cmd_levels,
            params={
                "reward_term_name": "track_linear_velocity",
            },
        ),
    }

    ##
    # Assemble and return
    ##

    cfg = ManagerBasedRlEnvCfg(
        capture_terminal_observations=True,
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=replace(ROUGH_TERRAINS_CFG, curriculum=True),
                max_init_terrain_level=5,
            ),
            entities={"robot": get_go2_robot_cfg()},
            sensors=(terrain_scan, feet_ground_cfg, base_ground_cfg),
            num_envs=1 if play else 4096,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.5,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=1500,
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
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Go2 HIM on flat terrain using the same training settings."""
    cfg = rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
        twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
        twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    return cfg
