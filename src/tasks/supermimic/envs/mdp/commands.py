# Copyright (c) 2026-2027 zh
"""HIM velocity sampling and heading feedback shared by real and imagined task state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from ...world_model.state_spec import Memory, Physical, TaskState
from ..sim_context import SimContext
from ..state_io import StateIO
from .curriculums import TerrainCurriculum
from .observations import OBS_DIM
from .terrains import PlaneSurface, TerrainMap, TerrainSurface

if TYPE_CHECKING:
    from ...config.go2.env_cfg import EnvCfg


def sample_commands(cfg, yaw):
    """Sample forward overrides before heading feedback and standing-command zeroing."""
    bounds = yaw.new_tensor(cfg.initial_ranges).expand(yaw.shape[0], -1, -1)
    random = torch.rand_like(bounds[..., 0])
    command = bounds[..., 0] + random * (bounds[..., 1] - bounds[..., 0])
    heading = cfg.heading_range[0] + torch.rand_like(yaw) * (cfg.heading_range[1] - cfg.heading_range[0])
    standing = torch.rand_like(yaw) <= cfg.rel_standing_envs
    forward = torch.rand_like(yaw) <= cfg.rel_forward_envs
    command = torch.stack(
        (
            torch.where(forward, command[:, 0].abs().clamp_min(0.3), command[:, 0]),
            torch.where(forward, torch.zeros_like(yaw), command[:, 1]),
            command[:, 2],
        ),
        -1,
    )
    lower, upper = cfg.resampling_time
    timer = lower + (upper - lower) * torch.rand_like(yaw)
    return command, timer, heading, standing, forward


def heading_command(cfg, command, yaw, heading, standing):
    """Apply heading feedback, including angular commands for forward samples."""
    error = wrap_to_pi(heading - yaw)
    angular = (cfg.heading_control_stiffness * error).clamp(*cfg.initial_ranges[2])
    result = torch.cat((command[:, :2], angular[:, None]), -1)
    return torch.where(standing[:, None].bool(), torch.zeros_like(result), result)


def advance_commands(cfg, state, physical, dt):
    """Refresh commands at the final heading while retaining fixed command overrides."""
    memory = Memory.split(state.memory)
    yaw = Physical.split(physical).pose[:, 2]
    sampled, timer, heading, standing, forward = sample_commands(cfg, yaw)
    expired = memory.command_timer.squeeze(-1) - dt <= 0.0
    command = torch.where(expired[:, None], sampled, state.command)
    timer = torch.where(expired, timer, memory.command_timer.squeeze(-1) - dt)
    heading = torch.where(expired, heading, memory.heading_target.squeeze(-1))
    standing = torch.where(expired, standing, memory.standing.squeeze(-1).bool())
    forward = torch.where(expired, forward, memory.forward.squeeze(-1).bool())
    command = heading_command(cfg, command, yaw, heading, standing)
    command = torch.where(memory.command_override.bool(), state.command, command)
    return command, timer, heading, standing, forward


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
    env_cfg: EnvCfg
    terrain_map: TerrainMap
    play: bool = False
    no_terminations: bool = False
    resampling_time_range: tuple[float, float] = (0.0, 0.0)

    def build(self, env):
        return VelocityCommand(self, env)


class VelocityCommand(CommandTerm):
    """Keep task memory while the native environment owns physics and lifecycle."""

    def __init__(self, cfg: VelocityCommandCfg, env):
        from ...world_model.heads import AnalyticHeads

        super().__init__(cfg, env)
        self.env_cfg = cfg.env_cfg
        self.robot = env.scene["robot"]
        self.map = cfg.terrain_map
        terrain = env.scene.terrain
        self.map.bind(terrain.terrain_origins)
        self.surface = (
            PlaneSurface(device=self.device)
            if self.map.plane
            else TerrainSurface.from_model(env.sim.mj_model, self.device, terrain_map=self.map)
        )
        self.model = AnalyticHeads(self.env_cfg, self.robot, self.surface, device=self.device)
        self.state_io = StateIO(env, self.model.spec.surface_height)
        self.sim_context = SimContext(env)
        spec = self.model.spec
        self.state = TaskState(
            torch.zeros(self.num_envs, spec.delta_dim, device=self.device),
            torch.zeros(self.num_envs, 42, device=self.device),
            torch.zeros(self.num_envs, 3, device=self.device),
            torch.zeros(self.num_envs, spec.xi_dim, device=self.device),
        )
        self.final_state = self.state
        # Observation-manager shape probes only read this buffer and consume no RNG.
        self.frame = torch.zeros(self.num_envs, OBS_DIM, device=self.device)
        self.command_steps = 0
        self.last_metrics = {}
        self.reward_terms = {}
        self._reset_ids = None
        self._commands = None
        self._enabled = None
        self._sliders = []
        self._gui_command = None

    def initialize_terrain(self):
        """Allocate curriculum after startup randomization, preserving its RNG order."""
        self.terrain_curriculum = TerrainCurriculum(
            self.map,
            self.num_envs,
            enabled=self.env_cfg.terrain_curriculum,
            device=self.device,
        )
        if not self.map.plane:
            self.terrain_curriculum.family_ids.copy_(self._env.scene.terrain.terrain_types)

    @property
    def command(self):
        return self.state.command

    @torch.no_grad()
    def set_commands(self, commands=None):
        """Fix commands for all environments, or restore command sampling with None."""
        if commands is None:
            self._commands = None
            Memory.split(self.state.memory).command_override.zero_()
        else:
            values = torch.as_tensor(commands, dtype=torch.float32, device=self.device)
            self._commands = values.expand(self.num_envs, -1).clone()
            self.state.command.copy_(self._commands)
            Memory.split(self.state.memory).command_override.fill_(1)

    def reset_origins(self, ids):
        if self.map.plane:
            return
        curriculum = self.terrain_curriculum
        if self.cfg.play:
            curriculum.family_ids[ids] = torch.randint(self.map.cols, (ids.numel(),), device=self.device)
            curriculum.levels[ids] = torch.randint(self.map.rows, (ids.numel(),), device=self.device)
        terrain = self._env.scene.terrain
        terrain.env_origins[ids] = curriculum.origins(ids)
        terrain.terrain_levels[ids] = curriculum.levels[ids]
        terrain.terrain_types[ids] = curriculum.family_ids[ids]

    def reset(self, env_ids):
        # Reset events have written qpos/qvel, but native forward() has not run yet.
        self._reset_ids = env_ids
        return {}

    @torch.no_grad()
    def compute(self, dt, env_ids=None):
        del dt
        if self._reset_ids is not None:
            ids = self._reset_ids
            reset = self.model.initialize_state(self.state_io.read()[ids], self.sim_context.read(ids))
            if self._commands is not None:
                reset.command.copy_(self._commands[ids])
                Memory.split(reset.memory).command_override.fill_(1)
            for name in ("physical", "memory", "command", "context"):
                getattr(self.state, name)[ids] = getattr(reset, name)
            self.terrain_curriculum.reset(ids, self.robot.data.root_link_pos_w[ids, :2])
            self._reset_ids = None
        if env_ids is None:
            self.frame = self.model.observe(self.state, noisy=True)
        else:
            self.frame[env_ids] = self.model.observe(
                TaskState(
                    *(getattr(self.state, name)[env_ids] for name in ("physical", "memory", "command", "context"))
                ),
                noisy=True,
            )
        self.last_metrics.update(self._env.extras["log"])
        self.last_metrics["task/terrain_level"] = self.terrain_curriculum.levels.float().mean()
        self._env.extras["log"].update(self.last_metrics)

    @torch.no_grad()
    def prepare_step(self):
        """Capture fresh terminal features before native termination and auto-reset."""
        env = self._env
        self._update_gui_commands()
        # Native reward/termination callbacks precede its final forward(). Task
        # coordinates need post-integration kinematics, including terminal rows.
        env.sim.forward()
        before = self.state
        sensor = env.scene["feet_ground_contact"].data
        action = env.action_manager.action
        final = self.model.advance(
            before,
            action,
            self.state_io.read(),
            contact_times=(sensor.current_air_time, sensor.current_contact_time),
        )
        if self._commands is not None:
            final.command.copy_(self._commands)
        terminated = (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if self.cfg.no_terminations
            else self.model.hard_termination(final)
        )
        self.final_state = final
        self.state = final.detached_clone()
        self.reward_terms = self.model.reward_terms(before, action, final)
        if not self.cfg.play:
            self.command_steps += 1
            self.terrain_curriculum.control_steps = env.common_step_counter
        done = terminated.clone()
        if not self.cfg.play and not self.cfg.no_terminations:
            done |= env.episode_length_buf >= env.max_episode_length
        # Settle using the executed command before integrating the terminal step.
        self.terrain_curriculum.update(env, done.nonzero().flatten(), before.command)
        self.terrain_curriculum.advance(before.command, self.robot.data.heading_w, env.step_dt)
        velocity = self.model.tracking_velocity(final)
        self.last_metrics = {}
        for name, value in self.reward_terms.items():
            mean = value.mean()
            weight = getattr(self.env_cfg.reward.scales, name)
            self.last_metrics[f"task/reward_raw/{name}"] = mean
            self.last_metrics[f"task/reward/{name}"] = env.step_dt * weight * mean
            self.last_metrics[f"Reward_Rate/{name}"] = weight * mean
        self.last_metrics.update(
            {
                "task/command_stage_iteration": float(self.command_steps),
                "task/tracking_linear_mse": (velocity[:, :2] - before.command[:, :2]).square().sum(-1).mean(),
                "task/tracking_angular_mse": (velocity[:, 2] - before.command[:, 2]).square().mean(),
                "task/action_rms": action.square().mean().sqrt(),
                "task/base_height": Physical.split(final.physical).height.mean(),
                "task/fast_yaw": (Physical.split(final.physical).angular_body[:, 2].abs() > 3.0).float().mean(),
            }
        )
        return terminated

    def curriculum_state_dict(self):
        return {"command": {"control_steps": self.command_steps}, "terrain": self.terrain_curriculum.state_dict()}

    def load_curriculum_state_dict(self, state):
        self.command_steps = state["command"]["control_steps"]
        self.terrain_curriculum.load_state_dict(state["terrain"])
        self._env.common_step_counter = self.command_steps

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
        del name, get_env_idx, on_change, request_action
        with server.gui.add_folder("Velocity control"):
            self._enabled = server.gui.add_checkbox("Manual commands (all environments)", initial_value=False)
            axes = ("Forward velocity vx (m/s)", "Lateral velocity vy (m/s)", "Yaw rate wz (rad/s)")
            for label, bounds in zip(axes, self.env_cfg.commands.initial_ranges, strict=True):
                limit = max(abs(bounds[0]), abs(bounds[1]))
                maximum = server.gui.add_slider(f"Range: {label}", min=0.1, max=10.0, step=0.1, initial_value=limit)
                slider = server.gui.add_slider(label, min=-limit, max=limit, step=0.05, initial_value=0.0)

                @maximum.on_update
                def _(_event, target=slider, source=maximum):
                    target.min, target.max = -source.value, source.value

                self._sliders.append(slider)
            zero = server.gui.add_button("Zero velocity")

            @zero.on_click
            def _(_event):
                for slider in self._sliders:
                    slider.value = 0.0

    def _update_gui_commands(self):
        command = (
            tuple(slider.value for slider in self._sliders)
            if self._enabled is not None and self._enabled.value
            else None
        )
        if command != self._gui_command:
            self.set_commands(command)
            self._gui_command = command

    # The analytic task owns resampling and memory, so native timers do not draw RNG.
    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        pass

    def _update_command(self, env_ids):
        pass


def observation(env):
    return env.command_manager.get_term("velocity").frame


def task_termination(env):
    return env.command_manager.get_term("velocity").prepare_step()


def reward_term(env, name):
    return env.command_manager.get_term("velocity").reward_terms[name]


def reset_origins(env, env_ids):
    env.command_manager.get_term("velocity").reset_origins(env_ids)


def initialize_terrain(env, env_ids):
    del env_ids
    env.command_manager.get_term("velocity").initialize_terrain()
