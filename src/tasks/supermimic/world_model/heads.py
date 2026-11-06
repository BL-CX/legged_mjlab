# Copyright (c) 2026-2027 zh
"""Analytic observations, commands, rewards, and deterministic task memory."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

import torch
from torch import Tensor

from ..envs.mdp.commands import advance_commands, heading_command, sample_commands
from ..envs.mdp.observations import analytic_observation, privileged_observation
from ..envs.mdp.reference_rewards import reference_reward_terms
from ..envs.mdp.terminations import hard_termination, soft_termination
from .state_spec import (
    AIR_TIME,
    COMMAND_OVERRIDE,
    COMMAND_TIMER,
    CONTACT,
    CONTACT_FOUND,
    CONTACT_TIME,
    ELAPSED_TIME,
    FORWARD,
    HEADING_TARGET,
    MEMORY_DIM,
    OBSERVATION_TERM_DIMS,
    PREVIOUS_ACTION,
    STANDING,
    YAW,
    StateSpec,
    TaskState,
    body_planar_velocity,
    tracking_velocity,
)


class AnalyticHeads:
    """Task mathematics in the native robot's joint and foot order."""

    observation_term_dims = OBSERVATION_TERM_DIMS

    def __init__(self, cfg, robot, surface, device="cpu"):
        self.cfg = deepcopy(cfg)
        self.device = torch.device(device)
        self.dt = cfg.env.timestep * cfg.env.decimation
        self.default_joint_pos = robot.data.default_joint_pos[0].clone()
        self.joint_limits = robot.data.soft_joint_pos_limits[0].clone()
        self.spec = StateSpec(
            surface,
            self.device,
            joint_names=robot.joint_names,
            history_length=cfg.env.num_obs_history,
            semantic_hash=_task_semantic_hash(cfg, surface.geometry_hash, self.default_joint_pos, self.joint_limits),
        )
        self.hip_joint_ids = [index for index, name in enumerate(self.spec.joint_names) if name.endswith("_hip_joint")]
        self.posture_std = torch.tensor(
            [
                [getattr(cfg.reward.posture, name)[joint.split("_")[1]] for joint in self.spec.joint_names]
                for name in ("std_standing", "std_walking", "std_running")
            ],
            device=self.device,
        )
        noise = cfg.noise.noise_scales
        self.noise_amplitude = (
            self.default_joint_pos.new_tensor(
                (noise.base_ang_vel,) * 3
                + (noise.projected_gravity,) * 3
                + (noise.joint_pos,) * self.spec.act_dim
                + (noise.joint_vel,) * self.spec.act_dim
                + (0.0,) * (self.spec.act_dim + self.spec.command_dim + 2)
            )
            * cfg.noise.noise_level
        )

    def tracking_velocity(self, state: TaskState) -> Tensor:
        velocity = tracking_velocity(state.physical)
        return torch.cat((body_planar_velocity(state.physical), velocity[:, 2:]), -1)

    def privileged(self, state: TaskState) -> Tensor:
        return privileged_observation(self, state)

    def observe(self, state: TaskState, *, noisy: bool = False) -> Tensor:
        return analytic_observation(self, state, noisy=noisy)

    def initialize_state(self, physical: Tensor, context: Tensor) -> TaskState:
        """Reset commands and deterministic memory, starting contact timers at zero."""
        command, timer, heading, standing, forward = sample_commands(self.cfg.commands, context, physical[:, YAW])
        memory = physical.new_zeros((physical.shape[0], MEMORY_DIM))
        memory[:, COMMAND_TIMER] = timer
        memory[:, HEADING_TARGET] = heading
        memory[:, STANDING] = standing
        memory[:, FORWARD] = forward
        command = heading_command(self.cfg.commands, command, physical[:, YAW], heading, standing)
        return TaskState(physical, memory, command, context)

    def advance(self, state: TaskState, action: Tensor, next_physical: Tensor, *, contact_times=None) -> TaskState:
        """Use real sensor contact timers or probability-weighted imagined durations."""
        clock = state.memory[:, ELAPSED_TIME] + self.dt
        command, timer, heading, standing, forward = advance_commands(self.cfg.commands, state, next_physical, self.dt)
        if contact_times is None:
            contact = next_physical[:, CONTACT_FOUND]
            previous = state.memory[:, CONTACT]
            # Maximal persistence coupling avoids false transitions at constant contact probability.
            contact_stay = torch.minimum(previous, contact) / previous.clamp_min(1e-6)
            air_stay = torch.minimum(1 - previous, 1 - contact) / (1 - previous).clamp_min(1e-6)
            air_time = air_stay * state.memory[:, AIR_TIME] + (1 - contact) * self.dt
            contact_time = contact_stay * state.memory[:, CONTACT_TIME] + contact * self.dt
        else:
            air_time, contact_time = contact_times
            contact = (contact_time > 0).to(next_physical.dtype)
        memory = torch.cat(
            (
                action,
                clock[:, None],
                timer[:, None],
                heading[:, None],
                standing.to(action.dtype)[:, None],
                forward.to(action.dtype)[:, None],
                contact,
                air_time,
                contact_time,
                state.memory[:, COMMAND_OVERRIDE, None],
                state.memory[:, PREVIOUS_ACTION],
            ),
            -1,
        )
        return TaskState(next_physical, memory, command, state.context)

    def reward_terms(self, state, action, next_state, failure):
        return reference_reward_terms(self, state, action, next_state, failure)

    def reward(self, state: TaskState, action: Tensor, next_state: TaskState, failure: Tensor) -> Tensor:
        return self.dt * sum(
            getattr(self.cfg.reward.scales, name) * value
            for name, value in self.reward_terms(state, action, next_state, failure).items()
        )

    def hard_termination(self, state: TaskState) -> Tensor:
        return hard_termination(state, self.cfg.reward.fall_angle)

    def soft_termination(self, state: TaskState) -> Tensor:
        return soft_termination(state, self.cfg.reward.fall_angle, self.cfg.reward.fall_softness_deg)


def _task_semantic_hash(cfg, geometry_hash: str, default_joint_pos: Tensor, joint_limits: Tensor) -> str:
    """Identify task coordinates, native physics, observations, and reward meaning."""
    from ..config import robot_settings

    task = asdict(cfg)
    for name in ("num_envs", "seed", "field_cache_dir"):
        task["env"].pop(name)
    payload = {
        "task": task,
        "robot": robot_settings(),
        "geometry": geometry_hash,
        "default_joint_pos": default_joint_pos.cpu().tolist(),
        "joint_limits": joint_limits.cpu().tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
