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
    effort_limit=23.7,
    armature=0.01,
    delay_min_lag=0,
    delay_max_lag=4,
    delay_hold_prob=0.5,
    delay_update_period=10,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*thigh_.*",),
    stiffness=20.0,
    damping=1.0,
    effort_limit=23.7,
    armature=0.01,
    delay_min_lag=0,
    delay_max_lag=4,
    delay_hold_prob=0.5,
    delay_update_period=10,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
    target_names_expr=(".*calf_.*",),
    stiffness=40.0,
    damping=2.0,
    effort_limit=45.43,
    armature=0.01,
    delay_min_lag=0,
    delay_max_lag=4,
    delay_hold_prob=0.5,
    delay_update_period=10,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.29),
    joint_pos={
        ".*thigh_joint": 0.9,
        ".*calf_joint": -1.8,
        ".*R_hip_joint": -0.1,
        ".*L_hip_joint": 0.1,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"
_collision_regex = r".*_collision\d*"

# Enable ground and self contacts; feet use three-dimensional frictional contact.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(_collision_regex,),
    condim={_foot_regex: 3, _collision_regex: 1},
    priority={_foot_regex: 1, ".*": 0},
    friction={_foot_regex: (1.0, 5e-3, 5e-4)},
    solref=(0.01, 1.0),
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


def get_go2_robot_cfg() -> EntityCfg:
    """Get a fresh Go2 robot configuration instance."""
    return EntityCfg(
        init_state=deepcopy(INIT_STATE),
        collisions=(deepcopy(FULL_COLLISION),),
        spec_fn=get_spec,
        articulation=deepcopy(GO2_ARTICULATION),
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_go2_robot_cfg())

    viewer.launch(robot.spec.compile())
