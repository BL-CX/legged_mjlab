"""Performance-based Go2 terrain progression."""

import numpy as np
import torch
from mjlab.terrains.terrain_entity import _proportional_counts


class TerrainCurriculum:
    """Apply Go2 HIM terrain progression to the existing map and starting levels."""

    def __init__(self, terrain_map, num_envs, *, enabled=True, device="cpu"):
        self.map = terrain_map
        self.enabled = enabled and not terrain_map.plane
        self.device = torch.device(device)
        self.control_steps = 0
        # MjLab assigns contiguous environment groups to columns.
        counts = torch.from_numpy(_proportional_counts(num_envs, np.asarray(terrain_map.proportions))).to(device)
        self.family_ids = torch.repeat_interleave(torch.arange(terrain_map.cols, device=device), counts)
        self.levels = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.episode_start_pos_w = torch.zeros(num_envs, 2, device=device)
        self.commanded_displacement_w = torch.zeros_like(self.episode_start_pos_w)
        if not self.enabled and not terrain_map.plane:
            self.levels = torch.randint(terrain_map.rows, (num_envs,), device=device)

    def reset(self, ids, positions):
        """Start an episode at its actual spawn position without changing difficulty."""
        self.episode_start_pos_w[ids] = positions
        self.commanded_displacement_w[ids] = 0

    def advance(self, command, yaw, dt):
        """Accumulate executed commands as world-frame displacement."""
        if not self.enabled:
            return
        vx, vy = command[:, :2].unbind(-1)
        velocity = torch.stack((yaw.cos() * vx - yaw.sin() * vy, yaw.sin() * vx + yaw.cos() * vy), -1)
        self.commanded_displacement_w.add_(velocity * dt)

    def update(self, env, ids, command):
        """Settle finished episodes before integrating the current terminal-step command."""
        if not self.enabled or ids.numel() == 0:
            return
        terrain = env.scene.terrain
        position = env.scene["robot"].data.root_link_pos_w[ids, :2]
        distance = (position - self.episode_start_pos_w[ids]).norm(dim=-1)
        time_left = env.max_episode_length_s - env.episode_length_buf[ids] * env.step_dt
        commanded_distance = self.commanded_displacement_w[ids].norm(dim=-1) + command[ids, :2].norm(
            dim=-1
        ) * time_left.clamp_min(0)
        terrain_distance = (position - env.scene.env_origins[ids, :2]).norm(dim=-1)
        move_up = (terrain_distance > self.map.size[0] / 2) & (commanded_distance >= 1.0)
        move_down = (distance < commanded_distance * 0.5) & ~move_up
        has_episode = (env.episode_length_buf[ids] > 0) & (env.common_step_counter > 0)
        move_up &= has_episode
        move_down &= has_episode
        # Use the native API for random overflow levels and clamping at level zero.
        terrain.terrain_levels[ids] = self.levels[ids]
        terrain.update_env_origins(ids, move_up, move_down)
        self.levels[ids] = terrain.terrain_levels[ids]

    def origins(self, ids):
        return self.map.spawn_origins(self.family_ids[ids], self.levels[ids])

    def state_dict(self):
        return {
            "family_ids": self.family_ids.clone(),
            "levels": self.levels.clone(),
            "control_steps": self.control_steps,
        }

    def load_state_dict(self, state):
        self.family_ids.copy_(state["family_ids"].to(self.device))
        self.levels.copy_(state["levels"].to(self.device))
        self.control_steps = state["control_steps"]
