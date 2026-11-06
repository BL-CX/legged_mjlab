"""Read instantaneous Go2 physical features in the task state order."""

import torch
from torch import Tensor

from ..kinematics.rotations import yaw_rotate_inverse


class StateIO:
    """Bind simulator indices and read physical features after forward()."""

    def __init__(self, env, surface_height):
        self.env = env
        self.surface_height = surface_height
        robot = env.scene["robot"]
        foot_names = tuple(dict.fromkeys(name.split("_")[0] for name in robot.joint_names))
        self.hip_ids, _ = robot.find_bodies(tuple(f"{leg}_thigh" for leg in foot_names), preserve_order=True)
        self.knee_ids, _ = robot.find_bodies(tuple(f"{leg}_calf" for leg in foot_names), preserve_order=True)
        self.foot_ids, _ = robot.find_sites(foot_names, preserve_order=True)

    def read(self) -> Tensor:
        """Read instantaneous features; use the largest leg-force norm across history."""
        env = self.env
        data = env.scene["robot"].data
        sensor = env.scene["feet_ground_contact"].data
        position, yaw = data.root_link_pos_w, data.heading_w
        limbs = (
            data.body_link_pos_w[:, self.hip_ids],
            data.body_link_pos_w[:, self.knee_ids],
            data.site_pos_w[:, self.foot_ids],
        )
        relative = [yaw_rotate_inverse(points - position[:, None], yaw[:, None]).flatten(1) for points in limbs]
        height = position[:, 2:3] - self.surface_height(position[:, :2]).unsqueeze(-1)
        history = env.scene["leg_ground_contact"].data.force_history
        leg_force = history.norm(dim=-1).amax(dim=(1, 2)).unsqueeze(-1)
        # Heading rotation preserves the squared angular-momentum norm without requiring absolute yaw.
        momentum = yaw_rotate_inverse(env.scene["robot/root_angmom"].data, yaw)
        physical = torch.cat(
            (
                height,
                data.projected_gravity_b,
                yaw_rotate_inverse(data.root_link_lin_vel_w, yaw),
                data.root_link_ang_vel_b,
                data.joint_pos - data.default_joint_pos,
                data.joint_vel,
                *relative,
                position[:, :2],
                yaw.unsqueeze(-1),
                data.joint_acc,
                data.site_lin_vel_w[:, self.foot_ids, :2].flatten(1),
                (sensor.found > 0).to(position.dtype),
                sensor.force.flatten(1),
                data.actuator_force,
                leg_force,
                momentum,
            ),
            -1,
        )
        return physical
