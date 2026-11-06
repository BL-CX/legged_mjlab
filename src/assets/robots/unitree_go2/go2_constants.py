"""Unitree Go2 constants."""

from copy import deepcopy
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH

##
# MJCF and assets.
##

GO2_XML: Path = SRC_PATH / "assets" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
assert GO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(GO2_XML))


##
# Actuator config.
##

GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
    target_names_expr=(".*hip_.*",),
    stiffness=20.0,
    damping=1.0,
    effort_limit=23.5,
    armature=0.01,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*thigh_.*",),
    stiffness=20.0,
    damping=1.0,
    effort_limit=23.5,
    armature=0.01,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
    target_names_expr=(".*calf_.*",),
    stiffness=40.0,
    damping=2.0,
    effort_limit=45,
    armature=0.02,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.40),
    joint_pos={
        ".*thigh_joint": 0.9,
        ".*calf_joint": -1.8,
        ".*R_hip_joint": 0.1,
        ".*L_hip_joint": -0.1,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
    solimp=(0.9, 0.95, 0.023),
)

# This enables collisions with the environment and between robot links.
# MuJoCo's parent filter still excludes directly connected parent-child bodies.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_foot_regex: 3, ".*_collision": 1},
    priority={_foot_regex: 1, ".*": 0},
    friction={_foot_regex: (0.6,)},
    solimp={_foot_regex: (0.9, 0.95, 0.023)},
    contype=1,
    conaffinity=1,
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        GO2_ACTUATOR_HIP,
        GO2_ACTUATOR_THIGH,
        GO2_ACTUATOR_CALF,
    ),
    soft_joint_pos_limit_factor=0.9,
)


DEFAULT_CFG = EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
)

# Delayed drives and the lower stance used for velocity control.
DELAYED_CFG = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.29),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.8,
        },
        joint_vel={".*": 0.0},
    ),
    collisions=(
        CollisionCfg(
            geom_names_expr=(r".*_collision\d*",),
            condim={_foot_regex: 3, r".*_collision\d*": 1},
            priority={_foot_regex: 1, ".*": 0},
            friction={_foot_regex: (1.0, 5e-3, 5e-4)},
            solref=(0.01, 1.0),
            contype=1,
            conaffinity=1,
        ),
    ),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=(
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_hip_joint",),
                stiffness=20.0,
                damping=1.0,
                effort_limit=23.7,
                armature=0.01,
                delay_min_lag=0,
                delay_max_lag=4,
                delay_hold_prob=0.5,
                delay_update_period=10,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_thigh_joint",),
                stiffness=20.0,
                damping=1.0,
                effort_limit=23.7,
                armature=0.01,
                delay_min_lag=0,
                delay_max_lag=4,
                delay_hold_prob=0.5,
                delay_update_period=10,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_calf_joint",),
                stiffness=40.0,
                damping=2.0,
                effort_limit=45.43,
                armature=0.01,
                delay_min_lag=0,
                delay_max_lag=4,
                delay_hold_prob=0.5,
                delay_update_period=10,
            ),
        ),
        soft_joint_pos_limit_factor=0.9,
    ),
)

ACTION_SCALE = {".*_hip_joint": 0.25, ".*_thigh_joint": 0.5, ".*_calf_joint": 0.5}


def get_go2_robot_cfg(cfg: EntityCfg = DEFAULT_CFG) -> EntityCfg:
    """Return an independent native Go2 configuration."""
    return deepcopy(cfg)


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_go2_robot_cfg())

    viewer.launch(robot.spec.compile())
