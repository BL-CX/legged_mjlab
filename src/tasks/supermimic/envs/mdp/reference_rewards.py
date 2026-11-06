# Copyright (c) 2026-2027 zh
"""Body-frame velocity tracking and the 21 HIM reward rates before control-period scaling."""

import torch

from ...world_model.state_spec import (
    ANGULAR_BODY,
    ANGULAR_MOMENTUM,
    CONTACT,
    CONTACT_FOUND,
    CONTACT_TIME,
    FOOT_CONTACT_FORCE,
    FOOT_VELOCITY_XY,
    GRAVITY,
    JOINT_ACCELERATION,
    JOINT_OFFSET,
    JOINT_TORQUE,
    JOINT_VELOCITY,
    LEG_CONTACT_FORCE,
    NUM_FEET,
    PREVIOUS_ACTION,
    PREVIOUS_PREVIOUS_ACTION,
    body_vertical_velocity,
)
from .gait_rewards import gait_reward_terms, posture_reward


def reference_reward_terms(task, state, action, next_state, failure):
    """Score the final state against the starting command for real and imagined transitions."""
    del failure
    cfg = task.cfg.reward
    physical = next_state.physical
    velocity = task.tracking_velocity(next_state)
    joint = physical[:, JOINT_OFFSET] + task.default_joint_pos
    lower, upper = task.joint_limits.unbind(-1)
    tilt_rate = physical[:, ANGULAR_BODY][:, :2].square().sum(-1)
    command_magnitude = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs()
    standing = command_magnitude <= cfg.stand_still_command_threshold
    foot_speed_squared = physical[:, FOOT_VELOCITY_XY].reshape(-1, NUM_FEET, 2).square().sum(-1)
    terms = {
        "track_linear_velocity": torch.exp(
            -(velocity[:, :2] - state.command[:, :2]).square().sum(-1) / cfg.tracking_linear_variance
        ),
        "track_angular_velocity": torch.exp(
            -(velocity[:, 2] - state.command[:, 2]).square() / cfg.tracking_angular_variance
        ),
        "action_rate_l2": (action - state.memory[:, PREVIOUS_ACTION]).square().sum(-1),
        "joint_pos_limits": ((lower - joint).clamp_min(0) + (joint - upper).clamp_min(0)).sum(-1),
        "body_ang_vel": tilt_rate,
        "lin_vel_z": body_vertical_velocity(physical).square(),
        "body_orientation_l2": physical[:, GRAVITY][:, :2].square().sum(-1),
        "smoothness": (action - 2 * state.memory[:, PREVIOUS_ACTION] + state.memory[:, PREVIOUS_PREVIOUS_ACTION])
        .square()
        .sum(-1),
        "stand_still": physical[:, JOINT_OFFSET].square().sum(-1) * standing,
        "stand_still_joint_vel_l2": physical[:, JOINT_VELOCITY].square().sum(-1) * standing,
        "foot_slip": (foot_speed_squared * physical[:, CONTACT_FOUND]).sum(-1)
        * (command_magnitude > cfg.foot_slip_command_threshold),
        "joint_acc_l2": physical[:, JOINT_ACCELERATION].square().sum(-1),
    }
    force = physical[:, FOOT_CONTACT_FORCE].reshape(-1, NUM_FEET, 3)
    terms["stumble"] = (force[..., :2].norm(dim=-1) > cfg.stumble_ratio * force[..., 2].abs()).any(dim=1).float()
    terms["joint_torques_l2"] = physical[:, JOINT_TORQUE].square().sum(-1)
    terms["leg_collision"] = (physical[:, LEG_CONTACT_FORCE] > cfg.leg_collision_force_threshold).any(dim=-1).float()
    terms["angular_momentum"] = physical[:, ANGULAR_MOMENTUM].square().sum(-1)
    terms["hip_pos"] = physical[:, JOINT_OFFSET][:, task.hip_joint_ids].square().sum(-1) * (
        (state.command[:, 1].abs() <= cfg.hip_pos_command_threshold)
        & (state.command[:, 2].abs() <= cfg.hip_pos_command_threshold)
    )
    contact = next_state.memory[:, CONTACT]
    # For soft contacts, recover conditional duration and weight landing force by contact probability.
    contact_time = next_state.memory[:, CONTACT_TIME] / contact.clamp_min(1e-6)
    first_contact = (contact_time > 0) & (contact_time < task.dt + 1e-6)
    force = physical[:, FOOT_CONTACT_FORCE].reshape(-1, NUM_FEET, 3)
    terms["soft_landing"] = (force.norm(dim=-1) * first_contact * contact).sum(-1) * (
        command_magnitude > cfg.soft_landing_command_threshold
    )
    terms.update(gait_reward_terms(task, state, next_state))
    terms["pose"] = posture_reward(task, state, next_state)
    return terms
