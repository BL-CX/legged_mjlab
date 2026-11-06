"""Analytic actor and critic observations shared by real and imagined states."""

import torch
from torch import Tensor

from ...world_model.state_spec import Context, Memory, Physical, TaskState
from .gait_rewards import gait_phase

OBS_TERMS = (
    ("velocity_command", 3),
    ("base_angular_velocity_body", 3),
    ("projected_gravity_body", 3),
    ("joint_position_offset_with_encoder_bias", 12),
    ("joint_velocity", 12),
    ("previous_unscaled_action", 12),
    ("gait_phase_sin_cos", 2),
)
OBS_DIM = sum(width for _, width in OBS_TERMS)
CRITIC_DIM = 262


def privileged_observation(task, state: TaskState) -> Tensor:
    """Build critic inputs with pose-detached terrain relief and differentiable base height."""
    physical = Physical.split(state.physical)
    memory = Memory.split(state.memory)
    context = Context.split(state.context)
    phase = 2 * torch.pi * gait_phase(state, task.dt, task.cfg.reward.gait.period)
    heading_error = memory.heading_target.squeeze(-1) - physical.pose[:, 2]
    task_features = torch.cat(
        (
            phase.sin()[:, None],
            phase.cos()[:, None],
            heading_error.sin()[:, None],
            heading_error.cos()[:, None],
            memory.standing,
            memory.forward,
            memory.contact,
            memory.air_time,
            memory.contact_time,
            memory.command_override,
            memory.previous_previous_action,
        ),
        -1,
    )
    # Detach terrain relief from pose gradients while retaining the base-height value gradient.
    scan = physical.height + (task.spec.terrain_scan(state.physical) - physical.height).detach()
    return torch.cat(
        (
            task.spec.physical_features(state.physical),
            memory.previous_action,
            state.command,
            memory.command_timer,
            scan,
            torch.cat((context.friction, context.com, context.encoder_bias), -1),
            task_features,
        ),
        -1,
    )


def analytic_observation(task, state: TaskState, *, noisy: bool = False) -> Tensor:
    """Build proprioception and gait phase, retaining the original actor-noise stream."""
    physical = Physical.split(state.physical)
    memory = Memory.split(state.memory)
    context = Context.split(state.context)
    observation = torch.cat(
        (
            state.command,
            physical.angular_body,
            physical.gravity,
            physical.joint_offset + context.encoder_bias,
            physical.joint_velocity,
            memory.previous_action,
        ),
        -1,
    )
    if noisy and task.cfg.noise.add_noise:
        # Keep the original 45-feature noise stream; phase features receive no noise.
        observation = (
            observation + (2 * torch.rand_like(observation) - 1) * task.noise_amplitude[: observation.shape[-1]]
        )
    phase = 2 * torch.pi * gait_phase(state, task.dt, task.cfg.reward.gait.period)
    moving = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs() > task.cfg.reward.gait.command_threshold
    phase_features = torch.stack((phase.sin(), phase.cos()), -1) * moving[:, None]
    return torch.cat((observation, phase_features), -1)
