"""Body-frame velocity tracking and the 21 HIM reward rates before control-period scaling."""

import torch

from ...world_model.state_spec import Memory, Physical, body_vertical_velocity
from .gait_rewards import gait_reward_terms, posture_reward


def reference_reward_terms(task, state, action, next_state):
    """Score the final state against the starting command for real and imagined transitions."""
    cfg = task.cfg.reward
    physical = Physical.split(next_state.physical)
    memory = Memory.split(state.memory)
    next_memory = Memory.split(next_state.memory)
    velocity = task.tracking_velocity(next_state)
    joint = physical.joint_offset + task.default_joint_pos
    lower, upper = task.joint_limits.unbind(-1)
    tilt_rate = physical.angular_body[:, :2].square().sum(-1)
    command_magnitude = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs()
    standing = command_magnitude <= cfg.stand_still_command_threshold
    foot_speed_squared = physical.foot_velocity_xy.reshape(-1, len(task.foot_names), 2).square().sum(-1)
    terms = {
        "track_linear_velocity": torch.exp(
            -(velocity[:, :2] - state.command[:, :2]).square().sum(-1) / cfg.tracking_linear_variance
        ),
        "track_angular_velocity": torch.exp(
            -(velocity[:, 2] - state.command[:, 2]).square() / cfg.tracking_angular_variance
        ),
        "action_rate_l2": (action - memory.previous_action).square().sum(-1),
        "joint_pos_limits": ((lower - joint).clamp_min(0) + (joint - upper).clamp_min(0)).sum(-1),
        "body_ang_vel": tilt_rate,
        "lin_vel_z": body_vertical_velocity(next_state.physical).square(),
        "body_orientation_l2": physical.gravity[:, :2].square().sum(-1),
        "smoothness": (action - 2 * memory.previous_action + memory.previous_previous_action).square().sum(-1),
        "stand_still": physical.joint_offset.square().sum(-1) * standing,
        "stand_still_joint_vel_l2": physical.joint_velocity.square().sum(-1) * standing,
        "foot_slip": (foot_speed_squared * physical.contact_found).sum(-1)
        * (command_magnitude > cfg.foot_slip_command_threshold),
        "joint_acc_l2": physical.joint_acceleration.square().sum(-1),
    }
    force = physical.foot_contact_force.reshape(-1, len(task.foot_names), 3)
    terms["stumble"] = (force[..., :2].norm(dim=-1) > cfg.stumble_ratio * force[..., 2].abs()).any(dim=1).float()
    terms["joint_torques_l2"] = physical.joint_torque.square().sum(-1)
    terms["leg_collision"] = (physical.leg_contact_force > cfg.leg_collision_force_threshold).any(dim=-1).float()
    terms["angular_momentum"] = physical.angular_momentum.square().sum(-1)
    terms["hip_pos"] = physical.joint_offset[:, task.hip_joint_ids].square().sum(-1) * (
        (state.command[:, 1].abs() <= cfg.hip_pos_command_threshold)
        & (state.command[:, 2].abs() <= cfg.hip_pos_command_threshold)
    )
    contact = next_memory.contact
    # For soft contacts, recover conditional duration and weight landing force by contact probability.
    contact_time = next_memory.contact_time / contact.clamp_min(1e-6)
    first_contact = (contact_time > 0) & (contact_time < task.dt + 1e-6)
    force = physical.foot_contact_force.reshape(-1, len(task.foot_names), 3)
    terms["soft_landing"] = (force.norm(dim=-1) * first_contact * contact).sum(-1) * (
        command_magnitude > cfg.soft_landing_command_threshold
    )
    terms.update(gait_reward_terms(task, state, next_state))
    terms["pose"] = posture_reward(task, state, next_state)
    return terms
