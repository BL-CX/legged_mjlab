# Copyright (c) 2026-2027 zh

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from .go2_constants import get_spec


def _make_actuator_cfg(control_cfg, joint_type: str) -> BuiltinPositionActuatorCfg:
    return BuiltinPositionActuatorCfg(
        target_names_expr=(f".*_{joint_type}_joint",),
        stiffness=control_cfg.stiffness[joint_type],
        damping=control_cfg.damping[joint_type],
        effort_limit=control_cfg.effort_limit[joint_type],
        armature=control_cfg.armature[joint_type],
        delay_min_lag=control_cfg.delay_min_lag,
        delay_max_lag=control_cfg.delay_max_lag,
        delay_hold_prob=control_cfg.delay_hold_prob,
        delay_update_period=control_cfg.delay_update_period,
    )


def _make_collision_cfg(collision_cfg) -> CollisionCfg:
    foot_regex = "^[FR][LR]_foot_collision$"
    # 本地小腿命名为 FL_calf1_collision，参考资产使用 FL_calf_collision1；
    # 此表达式同时覆盖两种碰撞几何命名。
    collision_regex = r".*_collision\d*"
    return CollisionCfg(
        geom_names_expr=(collision_regex,),
        condim={foot_regex: collision_cfg.foot_condim, collision_regex: collision_cfg.body_condim},
        priority={foot_regex: collision_cfg.foot_priority, ".*": 0},
        friction={foot_regex: tuple(collision_cfg.foot_friction)},
        solref=tuple(collision_cfg.solref),
        contype=1,
        conaffinity=1,
    )


def get_reference_go2_robot_cfg(cfg) -> EntityCfg:
    """复用本地模型及网格，每次创建独立的任务资产配置。"""
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(cfg.init_state.pos),
            joint_pos=dict(cfg.init_state.default_joint),
            joint_vel={".*": 0.0},
        ),
        collisions=(_make_collision_cfg(cfg.collision),),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(_make_actuator_cfg(cfg.control, name) for name in ("hip", "thigh", "calf")),
            soft_joint_pos_limit_factor=cfg.control.soft_joint_pos_limit_factor,
        ),
    )
