# Copyright (c) 2026-2027 zh
"""Analytic actor and critic observations shared by real and imagined states."""

import torch
from torch import Tensor

from ...world_model.state_spec import (
    AIR_TIME,
    ANGULAR_BODY,
    BASE_OBSERVATION_DIM,
    COMMAND_OVERRIDE,
    COMMAND_TIMER,
    CONTACT,
    CONTACT_TIME,
    CONTEXT_ENCODER_BIAS,
    FORWARD,
    GRAVITY,
    HEADING_TARGET,
    HEIGHT,
    JOINT_OFFSET,
    JOINT_VELOCITY,
    PREVIOUS_ACTION,
    PREVIOUS_PREVIOUS_ACTION,
    STANDING,
    TaskState,
)
from .gait_rewards import gait_phase


def privileged_observation(task, state: TaskState) -> Tensor:
    """Build critic inputs with pose-detached terrain relief and differentiable base height."""
    phase = 2 * torch.pi * gait_phase(state, task.dt, task.cfg.reward.gait.period)
    heading_error = state.memory[:, HEADING_TARGET] - task.spec.yaw(state)
    task_features = torch.cat(
        (
            phase.sin()[:, None],
            phase.cos()[:, None],
            heading_error.sin()[:, None],
            heading_error.cos()[:, None],
            state.memory[:, STANDING : FORWARD + 1],
            state.memory[:, CONTACT],
            state.memory[:, AIR_TIME],
            state.memory[:, CONTACT_TIME],
            state.memory[:, COMMAND_OVERRIDE, None],
            state.memory[:, PREVIOUS_PREVIOUS_ACTION],
        ),
        -1,
    )
    # Detach terrain relief from pose gradients while retaining the base-height value gradient.
    scan = state.physical[:, HEIGHT] + (task.spec.terrain_scan(state) - state.physical[:, HEIGHT]).detach()
    return torch.cat(
        (
            task.spec.physical_features(state),
            state.memory[:, PREVIOUS_ACTION],
            state.command,
            state.memory[:, COMMAND_TIMER, None],
            scan,
            task.spec.simulation_context(state),
            task_features,
        ),
        -1,
    )


def analytic_observation(task, state: TaskState, *, noisy: bool = False) -> Tensor:
    """Build proprioception and optional gait phase, retaining the original actor-noise stream."""
    physical = state.physical
    observation = torch.cat(
        (
            physical[:, ANGULAR_BODY],
            physical[:, GRAVITY],
            physical[:, JOINT_OFFSET] + state.context[:, CONTEXT_ENCODER_BIAS],
            physical[:, JOINT_VELOCITY],
            state.memory[:, PREVIOUS_ACTION],
            state.command,
        ),
        -1,
    )
    if noisy and task.cfg.noise.add_noise:
        # Keep the original 45-feature noise stream; phase features receive no noise.
        observation = (
            observation + (2 * torch.rand_like(observation) - 1) * task.noise_amplitude[:BASE_OBSERVATION_DIM]
        )
    phase = 2 * torch.pi * gait_phase(state, task.dt, task.cfg.reward.gait.period)
    moving = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs() > task.cfg.reward.gait.command_threshold
    phase_features = torch.stack((phase.sin(), phase.cos()), -1) * moving[:, None]
    return torch.cat((observation, phase_features), -1)
