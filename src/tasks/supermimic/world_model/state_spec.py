# Copyright (c) 2026-2027 zh
"""Quadruped state coordinates, packed trajectories, and world-model residuals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from ..kinematics.rotations import apply_pose_delta, pose_delta, yaw_rotate_inverse


@dataclass
class TaskState:
    """Feature-last physical state, deterministic task memory, commands, and context."""

    physical: Tensor
    memory: Tensor
    command: Tensor
    context: Tensor

    def detached_clone(self) -> TaskState:
        return TaskState(**{field.name: getattr(self, field.name).detach().clone() for field in fields(self)})


def _layout(fields):
    offset, slices = 0, {}
    for name, width in fields:
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices, offset


ACTION_DIM, NUM_FEET, COMMAND_DIM = 12, 4, 3
PHYSICAL_FIELDS = (
    ("height", 1),
    ("gravity", 3),
    ("velocity_heading", 3),
    ("angular_body", 3),
    ("joint_offset", ACTION_DIM),
    ("joint_velocity", ACTION_DIM),
    ("hips", NUM_FEET * 3),
    ("knees", NUM_FEET * 3),
    ("feet", NUM_FEET * 3),
    ("position_xy", 2),
    ("yaw", 1),
    ("joint_acceleration", ACTION_DIM),
    ("foot_velocity_xy", NUM_FEET * 2),
    ("contact_found", NUM_FEET),
    ("foot_contact_force", NUM_FEET * 3),
    ("joint_torque", ACTION_DIM),
    ("leg_contact_force", 1),
    ("angular_momentum", 3),
)
PHYSICAL, PHYSICAL_DIM = _layout(PHYSICAL_FIELDS)
HEIGHT = PHYSICAL["height"]
GRAVITY = PHYSICAL["gravity"]
VELOCITY_HEADING = PHYSICAL["velocity_heading"]
ANGULAR_BODY = PHYSICAL["angular_body"]
JOINT_OFFSET = PHYSICAL["joint_offset"]
JOINT_VELOCITY = PHYSICAL["joint_velocity"]
HIPS = PHYSICAL["hips"]
KNEES = PHYSICAL["knees"]
FEET = PHYSICAL["feet"]
POSITION_XY = PHYSICAL["position_xy"]
YAW = PHYSICAL["yaw"].start
POSE = slice(POSITION_XY.start, PHYSICAL["yaw"].stop)
ROBOT_STATE = slice(HEIGHT.start, POSE.start)
JOINT_ACCELERATION = PHYSICAL["joint_acceleration"]
FOOT_VELOCITY_XY = PHYSICAL["foot_velocity_xy"]
CONTACT_FOUND = PHYSICAL["contact_found"]
FOOT_CONTACT_FORCE = PHYSICAL["foot_contact_force"]
JOINT_TORQUE = PHYSICAL["joint_torque"]
LEG_CONTACT_FORCE = PHYSICAL["leg_contact_force"]
ANGULAR_MOMENTUM = PHYSICAL["angular_momentum"]
AUXILIARY = slice(JOINT_ACCELERATION.start, PHYSICAL_DIM)

MEMORY_FIELDS = (
    ("previous_action", ACTION_DIM),
    ("elapsed_time", 1),
    ("command_timer", 1),
    ("heading_target", 1),
    ("standing", 1),
    ("forward", 1),
    ("contact", NUM_FEET),
    ("air_time", NUM_FEET),
    ("contact_time", NUM_FEET),
    ("command_override", 1),
    ("previous_previous_action", ACTION_DIM),
)
MEMORY, MEMORY_DIM = _layout(MEMORY_FIELDS)
PREVIOUS_ACTION = MEMORY["previous_action"]
ELAPSED_TIME = MEMORY["elapsed_time"].start
COMMAND_TIMER = MEMORY["command_timer"].start
HEADING_TARGET = MEMORY["heading_target"].start
STANDING = MEMORY["standing"].start
FORWARD = MEMORY["forward"].start
CONTACT = MEMORY["contact"]
AIR_TIME = MEMORY["air_time"]
CONTACT_TIME = MEMORY["contact_time"]
COMMAND_OVERRIDE = MEMORY["command_override"].start
PREVIOUS_PREVIOUS_ACTION = MEMORY["previous_previous_action"]

CONTEXT_FIELDS = (
    ("friction", 1),
    ("encoder_bias", ACTION_DIM),
    ("com", 3),
    ("origin", 3),
    ("ranges", COMMAND_DIM * 2),
    ("zero_probability", 1),
)
CONTEXT, CONTEXT_DIM = _layout(CONTEXT_FIELDS)
CONTEXT_FRICTION = CONTEXT["friction"].start
CONTEXT_ENCODER_BIAS = CONTEXT["encoder_bias"]
CONTEXT_COM = CONTEXT["com"]
CONTEXT_ORIGIN = CONTEXT["origin"]
CONTEXT_RANGES = CONTEXT["ranges"]
CONTEXT_ZERO_PROBABILITY = CONTEXT["zero_probability"].start

OBS_TERMS = (
    ("base_angular_velocity_body", 3),
    ("projected_gravity_body", 3),
    ("joint_position_offset_with_encoder_bias", ACTION_DIM),
    ("joint_velocity", ACTION_DIM),
    ("previous_unscaled_action", ACTION_DIM),
    ("velocity_command", COMMAND_DIM),
    ("gait_phase_sin_cos", 2),
)
OBSERVATION_TERM_DIMS = tuple(width for _, width in OBS_TERMS)
OBSERVATION_DIM = sum(OBSERVATION_TERM_DIMS)
BASE_OBSERVATION_DIM = sum(OBSERVATION_TERM_DIMS[:-1])
SCAN_SHAPE = (11, 7)
SCAN_DIM = SCAN_SHAPE[0] * SCAN_SHAPE[1]
XI_DIM = sum(CONTEXT[name].stop - CONTEXT[name].start for name in ("friction", "com", "encoder_bias"))
TASK_FEATURE_DIM = 2 + 2 + 2 + NUM_FEET * 3 + 1 + ACTION_DIM
FEAT_DIM = PHYSICAL_DIM - (POSE.stop - POSE.start) + ACTION_DIM + SCAN_DIM
PRIVILEGED_DIM = FEAT_DIM + COMMAND_DIM + 1 + XI_DIM + TASK_FEATURE_DIM
STATE_DIM = PHYSICAL_DIM + MEMORY_DIM + COMMAND_DIM


def normalized_gravity(physical: Tensor) -> Tensor:
    """Normalize only for coordinate calculations; leave predicted physical state unchanged."""
    gravity = physical[..., GRAVITY]
    return gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def tracking_velocity(physical: Tensor) -> Tensor:
    """Return Appendix A.6 command coordinates: heading XY velocity and world yaw rate."""
    world_yaw_rate = -(normalized_gravity(physical) * physical[..., ANGULAR_BODY]).sum(-1, keepdim=True)
    return torch.cat((physical[..., VELOCITY_HEADING][..., :2], world_yaw_rate), -1)


def body_planar_velocity(physical: Tensor) -> Tensor:
    """Recover body XY velocity from heading velocity and unit gravity, including body tilt."""
    gravity = normalized_gravity(physical)
    gx, gy, gz = gravity.unbind(-1)
    cosine_pitch = gravity[..., 1:].norm(dim=-1).clamp_min(1e-6)
    velocity = physical[..., VELOCITY_HEADING]
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
    return (physical[..., VELOCITY_HEADING] * body_z_heading).sum(-1)


class StateSpec:
    """Features, terrain inputs, integration, and loss coordinates for the native robot."""

    physical_dim = PHYSICAL_DIM
    state_dim = STATE_DIM
    feat_dim = FEAT_DIM
    act_dim = ACTION_DIM
    delta_dim = PHYSICAL_DIM
    xi_dim = XI_DIM
    observation_dim = OBSERVATION_DIM
    privileged_dim = PRIVILEGED_DIM
    context_dim = CONTEXT_DIM
    memory_dim = MEMORY_DIM
    command_dim = COMMAND_DIM

    def __init__(self, surface, device, *, joint_names, history_length=5, semantic_hash=""):
        self.surface = surface
        self.joint_names = tuple(joint_names)
        self.foot_names = tuple(dict.fromkeys(name.split("_")[0] for name in self.joint_names))
        self.history_length = history_length
        payload = (
            PHYSICAL_FIELDS,
            MEMORY_FIELDS,
            CONTEXT_FIELDS,
            OBS_TERMS,
            self.joint_names,
            history_length,
            semantic_hash,
        )
        self.hash = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        # Reference forward-shifted scan grid, with X as the major axis.
        xs = -0.42 + 0.12 * torch.arange(SCAN_SHAPE[0], device=device)
        ys = -0.36 + 0.12 * torch.arange(SCAN_SHAPE[1], device=device)
        self.scan_points = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)

    def pack(self, state: TaskState) -> Tensor:
        """Keep physical state, deterministic memory, and commands in one tensor."""
        return torch.cat((state.physical, state.memory, state.command), -1)

    def unpack(self, state: Tensor, xi: Tensor) -> TaskState:
        """Read the packed state with per-environment simulation context."""
        context = xi.expand(*state.shape[:-1], CONTEXT_DIM)
        return TaskState(
            state[..., : self.physical_dim],
            state[..., self.physical_dim : self.physical_dim + MEMORY_DIM],
            state[..., self.physical_dim + MEMORY_DIM :],
            context,
        )

    def canonical(self, state: Tensor, recorded: Tensor | None = None) -> Tensor:
        """Build Go2 features; simulation context is supplied separately to dynamics."""
        current = self.unpack(state, state.new_zeros((*state.shape[:-1], CONTEXT_DIM)))
        if recorded is None:
            scan = self.terrain_scan(current)
        else:
            source = self.unpack(recorded, current.context)
            relief = (self.terrain_scan(source) - source.physical[:, HEIGHT]).half().float()
            scan = relief + current.physical[:, HEIGHT]
        return torch.cat((self.physical_features(current), current.memory[:, PREVIOUS_ACTION], scan), -1)

    @staticmethod
    def normalize_xi(xi: Tensor) -> Tensor:
        """Use unit scales for friction, COM offset, and encoder bias."""
        return torch.cat((xi[..., CONTEXT["friction"]], xi[..., CONTEXT_COM], xi[..., CONTEXT_ENCODER_BIAS]), -1)

    def error_vector(self, current: Tensor, predicted: Tensor, target: Tensor, recorded: Tensor, xi: Tensor) -> Tensor:
        """Retain task residual coordinates with fixed unit error scales."""
        return self.model_error(
            self.unpack(current, xi), self.unpack(predicted, xi), self.unpack(target, xi), self.unpack(recorded, xi)
        )

    def teacher_force(self, predicted: Tensor, recorded: Tensor) -> Tensor:
        """Feed back predicted physics and the recorded task memory and commands."""
        return torch.cat((predicted[..., : self.physical_dim], recorded[..., self.physical_dim :]), -1)

    def surface_height(self, xy: Tensor) -> Tensor:
        """Clamp height queries at the reference raster boundary."""
        return self.surface.height(xy)

    def base_height(self, state: TaskState) -> Tensor:
        return state.physical[:, HEIGHT.start]

    def yaw(self, state: TaskState) -> Tensor:
        return state.physical[:, YAW]

    def terrain_scan(self, state: TaskState) -> Tensor:
        xy, yaw = state.physical[:, POSITION_XY], self.yaw(state)
        x, y = self.scan_points.unbind(-1)
        points = (
            torch.stack(
                (yaw.cos()[:, None] * x - yaw.sin()[:, None] * y, yaw.sin()[:, None] * x + yaw.cos()[:, None] * y), -1
            )
            + xy[:, None, :]
        )
        relief = (self.surface_height(xy)[:, None] - self.surface_height(points)).clamp(-1.2, 1.2)
        return self.base_height(state)[:, None] + relief

    def physical_features(self, state: TaskState) -> Tensor:
        return torch.cat((state.physical[:, ROBOT_STATE], state.physical[:, AUXILIARY]), -1)

    @staticmethod
    def simulation_context(state: TaskState) -> Tensor:
        # Encoder bias affects both dynamics and value inputs through MjLab position actions.
        return torch.cat(
            (
                state.context[:, CONTEXT["friction"]],
                state.context[:, CONTEXT_COM],
                state.context[:, CONTEXT_ENCODER_BIAS],
            ),
            -1,
        )

    def integrate(self, state: TaskState, delta: Tensor) -> Tensor:
        return torch.cat(
            (
                state.physical[:, ROBOT_STATE] + delta[:, ROBOT_STATE],
                apply_pose_delta(state.physical[:, POSE], delta[:, POSE]),
                delta[:, AUXILIARY.start : CONTACT_FOUND.start],
                delta[:, CONTACT_FOUND].sigmoid(),
                delta[:, CONTACT_FOUND.stop :],
            ),
            -1,
        )

    def model_error(
        self, current: TaskState, predicted: TaskState, target: TaskState, recorded_current: TaskState
    ) -> Tensor:
        """Compare base-state deltas, auxiliary values, and predicted contact probabilities."""
        # Keep predicted yaw unwrapped so errors larger than pi receive the full penalty.
        predicted_delta = yaw_rotate_inverse(
            predicted.physical[:, POSE] - current.physical[:, POSE], self.yaw(current)
        )
        return torch.cat(
            (
                predicted.physical[:, ROBOT_STATE] - target.physical[:, ROBOT_STATE],
                predicted_delta - pose_delta(recorded_current.physical[:, POSE], target.physical[:, POSE]),
                predicted.physical[:, AUXILIARY] - target.physical[:, AUXILIARY],
            ),
            -1,
        )
