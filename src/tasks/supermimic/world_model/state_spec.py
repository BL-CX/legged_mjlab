# Copyright (c) 2026-2027 zh
"""Quadruped tensor views, canonical features, and world-model residuals."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import NamedTuple

import torch
from torch import Tensor

from ..kinematics.rotations import apply_pose_delta, pose_delta, yaw_rotate_inverse


class Physical(NamedTuple):
    """Views of the 125 physical coordinates; scalar fields retain their last axis."""

    height: Tensor
    gravity: Tensor
    velocity_heading: Tensor
    angular_body: Tensor
    joint_offset: Tensor
    joint_velocity: Tensor
    link_positions: Tensor
    pose: Tensor
    joint_acceleration: Tensor
    foot_velocity_xy: Tensor
    contact_found: Tensor
    foot_contact_force: Tensor
    joint_torque: Tensor
    leg_contact_force: Tensor
    angular_momentum: Tensor

    @staticmethod
    def split(value: Tensor) -> Physical:
        return Physical(*value.split((1, 3, 3, 3, 12, 12, 36, 3, 12, 8, 4, 12, 12, 1, 3), dim=-1))


class Memory(NamedTuple):
    """Views of deterministic command, action, and contact memory."""

    previous_action: Tensor
    elapsed_time: Tensor
    command_timer: Tensor
    heading_target: Tensor
    standing: Tensor
    forward: Tensor
    contact: Tensor
    air_time: Tensor
    contact_time: Tensor
    command_override: Tensor
    previous_previous_action: Tensor

    @staticmethod
    def split(value: Tensor) -> Memory:
        return Memory(*value.split((12, 1, 1, 1, 1, 1, 4, 4, 4, 1, 12), dim=-1))


class Context(NamedTuple):
    """Views of randomized simulator parameters, stored independently from state."""

    friction: Tensor
    encoder_bias: Tensor
    com: Tensor

    @staticmethod
    def split(value: Tensor) -> Context:
        return Context(*value.split((1, 12, 3), dim=-1))


@dataclass
class TaskState:
    """Feature-last physical state, deterministic task memory, commands, and context."""

    physical: Tensor
    memory: Tensor
    command: Tensor
    context: Tensor

    def detached_clone(self) -> TaskState:
        return TaskState(**{field.name: getattr(self, field.name).detach().clone() for field in fields(self)})


def normalized_gravity(physical: Tensor) -> Tensor:
    """Normalize only for coordinate calculations; leave predicted physical state unchanged."""
    gravity = Physical.split(physical).gravity
    return gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def tracking_velocity(physical: Tensor) -> Tensor:
    """Return Appendix A.6 command coordinates: heading XY velocity and world yaw rate."""
    current = Physical.split(physical)
    world_yaw_rate = -(normalized_gravity(physical) * current.angular_body).sum(-1, keepdim=True)
    return torch.cat((current.velocity_heading[..., :2], world_yaw_rate), -1)


def body_planar_velocity(physical: Tensor) -> Tensor:
    """Recover body XY velocity from heading velocity and unit gravity, including body tilt."""
    gravity = normalized_gravity(physical)
    gx, gy, gz = gravity.unbind(-1)
    cosine_pitch = gravity[..., 1:].norm(dim=-1).clamp_min(1e-6)
    velocity = Physical.split(physical).velocity_heading
    body_x = cosine_pitch * velocity[..., 0] - gx * velocity[..., 2]
    body_y = -gx * gy / cosine_pitch * velocity[..., 0] - gz / cosine_pitch * velocity[..., 1] - gy * velocity[..., 2]
    return torch.stack((body_x, body_y), -1)


def body_vertical_velocity(physical: Tensor) -> Tensor:
    """Recover body Z velocity from heading velocity and unit gravity."""
    gravity = normalized_gravity(physical)
    gx, gy, gz = gravity.unbind(-1)
    # Heading follows the horizontal body X projection; norm(g_yz) is nonnegative cos(pitch).
    cosine_pitch = gravity[..., 1:].norm(dim=-1).clamp_min(1e-6)
    body_z_heading = torch.stack((-gx * gz / cosine_pitch, gy / cosine_pitch, -gz), -1)
    return (Physical.split(physical).velocity_heading * body_z_heading).sum(-1)


class StateSpec:
    """Packed state coordinates, terrain features, integration, and loss coordinates."""

    state_dim = 170
    feat_dim = 211
    act_dim = 12
    delta_dim = 125
    xi_dim = 16

    def __init__(self, surface, device):
        self.surface = surface
        # Reference forward-shifted scan grid, with X as the major axis.
        xs = -0.42 + 0.12 * torch.arange(11, device=device)
        ys = -0.36 + 0.12 * torch.arange(7, device=device)
        self.scan_points = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)

    @staticmethod
    def split(state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return state.split((125, 42, 3), dim=-1)

    @staticmethod
    def pack(state: TaskState) -> Tensor:
        """Keep physical state, deterministic memory, and commands in one tensor."""
        return torch.cat((state.physical, state.memory, state.command), -1)

    def unpack(self, state: Tensor, xi: Tensor) -> TaskState:
        """Read the packed state with per-environment simulation parameters."""
        return TaskState(*self.split(state), xi.expand(*state.shape[:-1], self.xi_dim))

    def canonical(self, state: Tensor, recorded: Tensor | None = None) -> Tensor:
        """Build heading-invariant inputs, with recorded terrain relief during training."""
        physical, memory, _ = self.split(state)
        if recorded is None:
            scan = self.terrain_scan(physical)
        else:
            source, _, _ = self.split(recorded)
            relief = (self.terrain_scan(source) - Physical.split(source).height).half().float()
            scan = relief + Physical.split(physical).height
        return torch.cat((self.physical_features(physical), Memory.split(memory).previous_action, scan), -1)

    @staticmethod
    def normalize_xi(xi: Tensor) -> Tensor:
        """Use unit scales for friction, COM offset, and encoder bias."""
        context = Context.split(xi)
        return torch.cat((context.friction, context.com, context.encoder_bias), -1)

    @staticmethod
    def teacher_force(predicted: Tensor, recorded: Tensor) -> Tensor:
        """Feed back predicted physics and the recorded task memory and commands."""
        return torch.cat((predicted[..., :125], recorded[..., 125:]), -1)

    def surface_height(self, xy: Tensor) -> Tensor:
        """Clamp height queries at the reference raster boundary."""
        return self.surface.height(xy)

    def terrain_scan(self, physical: Tensor) -> Tensor:
        current = Physical.split(physical)
        xy, yaw = current.pose[..., :2], current.pose[..., 2]
        x, y = self.scan_points.unbind(-1)
        points = (
            torch.stack(
                (
                    yaw.cos()[..., None] * x - yaw.sin()[..., None] * y,
                    yaw.sin()[..., None] * x + yaw.cos()[..., None] * y,
                ),
                -1,
            )
            + xy[..., None, :]
        )
        relief = (self.surface_height(xy)[..., None] - self.surface_height(points)).clamp(-1.2, 1.2)
        return current.height + relief

    @staticmethod
    def physical_features(physical: Tensor) -> Tensor:
        robot, _, auxiliary = physical.split((70, 3, 52), dim=-1)
        return torch.cat((robot, auxiliary), -1)

    @staticmethod
    def integrate(physical: Tensor, delta: Tensor) -> Tensor:
        robot, pose, _ = physical.split((70, 3, 52), dim=-1)
        robot_delta, pose_change, auxiliary, contact, forces = delta.split((70, 3, 20, 4, 28), dim=-1)
        return torch.cat(
            (robot + robot_delta, apply_pose_delta(pose, pose_change), auxiliary, contact.sigmoid(), forces), -1
        )

    def error_vector(self, current: Tensor, predicted: Tensor, target: Tensor, recorded: Tensor) -> Tensor:
        """Compare base-state deltas, auxiliary values, and predicted contact probabilities."""
        current_physical, _, _ = self.split(current)
        predicted_physical, _, _ = self.split(predicted)
        target_physical, _, _ = self.split(target)
        recorded_physical, _, _ = self.split(recorded)
        _, current_pose, _ = current_physical.split((70, 3, 52), dim=-1)
        predicted_robot, predicted_pose, predicted_auxiliary = predicted_physical.split((70, 3, 52), dim=-1)
        target_robot, target_pose, target_auxiliary = target_physical.split((70, 3, 52), dim=-1)
        _, recorded_pose, _ = recorded_physical.split((70, 3, 52), dim=-1)
        # Keep predicted yaw unwrapped so errors larger than pi receive the full penalty.
        predicted_delta = yaw_rotate_inverse(predicted_pose - current_pose, current_pose[..., 2])
        return torch.cat(
            (
                predicted_robot - target_robot,
                predicted_delta - pose_delta(recorded_pose, target_pose),
                predicted_auxiliary - target_auxiliary,
            ),
            -1,
        )
