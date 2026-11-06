# Copyright (c) 2026-2027 zh
"""HIM contact rewards and their soft-contact approximation for imagination."""

import torch

from ...world_model.state_spec import (
    AIR_TIME,
    CONTACT,
    CONTACT_TIME,
    ELAPSED_TIME,
    JOINT_OFFSET,
)


def gait_phase(state, dt, period):
    """Match integer episode-step modulo to avoid floating-point clock drift."""
    steps = torch.round(state.memory[:, ELAPSED_TIME] / dt)
    return torch.remainder(steps, period / dt) / (period / dt)


def gait_reward_terms(task, state, next_state):
    """Score binary contacts directly and approximate soft contacts by expected matches and durations."""
    cfg = task.cfg.reward.gait
    phase = gait_phase(next_state, task.dt, cfg.period)
    offsets = phase.new_tensor([cfg.offsets[name] for name in task.spec.foot_names])
    desired = ((phase[:, None] + offsets) % 1.0 < cfg.stance_fraction).to(phase.dtype)
    contact = next_state.memory[:, CONTACT]
    active = (state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs() > cfg.command_threshold).to(phase.dtype)
    matches = desired * contact + (1 - desired) * (1 - contact)
    stance_duration = cfg.period * cfg.stance_fraction
    swing_duration = cfg.period * (1 - cfg.stance_fraction)
    # Recover conditional durations from probability-weighted memory; binary contacts remain unchanged.
    contact_duration = next_state.memory[:, CONTACT_TIME] / contact.clamp_min(1e-6)
    air_duration = next_state.memory[:, AIR_TIME] / (1 - contact).clamp_min(1e-6)
    stance_score = (1 - (contact_duration / stance_duration - 1).abs()).clamp_min(0)
    swing_score = (1 - (air_duration / swing_duration - 1).abs()).clamp_min(0)
    durations = desired * contact * stance_score + (1 - desired) * (1 - contact) * swing_score
    return {"foot_gait": matches.mean(-1) * active, "feet_air_time": durations.mean(-1) * active}


def posture_reward(task, state, next_state):
    """Use the reference joint-wise posture deviations and speed thresholds."""
    cfg = task.cfg.reward.posture
    magnitude = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs()
    standing = magnitude < cfg.walking_threshold
    walking = (magnitude >= cfg.walking_threshold) & (magnitude < cfg.running_threshold)
    std = (
        task.posture_std[0] * standing[:, None]
        + task.posture_std[1] * walking[:, None]
        + task.posture_std[2] * (magnitude >= cfg.running_threshold)[:, None]
    )
    return torch.exp(-(next_state.physical[:, JOINT_OFFSET].square() / std.square()).mean(-1))
