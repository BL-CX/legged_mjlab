"""Analytic observations, commands, rewards, and deterministic task memory."""

import torch
from torch import Tensor

from ..envs.mdp.commands import advance_commands, heading_command, sample_commands
from ..envs.mdp.observations import (
    OBS_TERMS,
    analytic_observation,
    privileged_observation,
)
from ..envs.mdp.reference_rewards import reference_reward_terms
from ..envs.mdp.terminations import hard_termination, soft_termination
from .state_spec import (
    Memory,
    Physical,
    StateSpec,
    TaskState,
    body_planar_velocity,
    tracking_velocity,
)


class AnalyticHeads:
    """Task mathematics in the native robot's joint and foot order."""

    observation_term_dims = tuple(width for _, width in OBS_TERMS)

    def __init__(
        self,
        cfg,
        robot,
        surface,
        *,
        dt: float,
        reward_scale: float,
        reward_weights: dict[str, float],
        device="cpu",
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dt = dt
        self.reward_scale = reward_scale
        self.reward_weights = reward_weights
        self.joint_names = tuple(robot.joint_names)
        self.foot_names = tuple(dict.fromkeys(name.split("_")[0] for name in self.joint_names))
        self.default_joint_pos = robot.data.default_joint_pos[0].clone()
        self.joint_limits = robot.data.soft_joint_pos_limits[0].clone()
        self.spec = StateSpec(surface, self.device)
        self.hip_joint_ids = [index for index, name in enumerate(self.joint_names) if name.endswith("_hip_joint")]
        self.posture_std = torch.tensor(
            [
                [getattr(cfg.reward.posture, name)[joint.split("_")[1]] for joint in self.joint_names]
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
                + (0.0,) * (self.spec.act_dim + 3 + 2)
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
        yaw = Physical.split(physical).pose[:, 2]
        command, timer, heading, standing, forward = sample_commands(self.cfg.commands, yaw)
        memory = physical.new_zeros((physical.shape[0], 42))
        fields = Memory.split(memory)
        fields.command_timer.copy_(timer[:, None])
        fields.heading_target.copy_(heading[:, None])
        fields.standing.copy_(standing[:, None])
        fields.forward.copy_(forward[:, None])
        command = heading_command(self.cfg.commands, command, yaw, heading, standing)
        return TaskState(physical, memory, command, context)

    def advance(
        self,
        state: TaskState,
        action: Tensor,
        next_physical: Tensor,
        *,
        contact_times=None,
    ) -> TaskState:
        """Use real sensor contact timers or probability-weighted imagined durations."""
        before = Memory.split(state.memory)
        clock = before.elapsed_time.squeeze(-1) + self.dt
        command, timer, heading, standing, forward = advance_commands(self.cfg.commands, state, next_physical, self.dt)
        if contact_times is None:
            contact = Physical.split(next_physical).contact_found
            previous = before.contact
            # Maximal persistence coupling avoids false transitions at constant contact probability.
            contact_stay = torch.minimum(previous, contact) / previous.clamp_min(1e-6)
            air_stay = torch.minimum(1 - previous, 1 - contact) / (1 - previous).clamp_min(1e-6)
            air_time = air_stay * before.air_time + (1 - contact) * self.dt
            contact_time = contact_stay * before.contact_time + contact * self.dt
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
                before.command_override,
                before.previous_action,
            ),
            -1,
        )
        return TaskState(next_physical, memory, command, state.context)

    def reward_terms(self, state, action, next_state):
        return reference_reward_terms(self, state, action, next_state)

    def reward(self, state: TaskState, action: Tensor, next_state: TaskState) -> Tensor:
        return self.reward_scale * sum(
            self.reward_weights.get(name, 0.0) * value
            for name, value in self.reward_terms(state, action, next_state).items()
        )

    def hard_termination(self, state: TaskState) -> Tensor:
        return hard_termination(state, self.cfg.reward.fall_angle)

    def soft_termination(self, state: TaskState) -> Tensor:
        return soft_termination(state, self.cfg.reward.fall_angle, self.cfg.reward.fall_softness_deg)
