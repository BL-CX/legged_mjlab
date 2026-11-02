"""Terrain-specific velocity sampling using MJLab's heading controller."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TerrainVelocityCommand(UniformVelocityCommand):
  """Reuse native heading updates, metrics and visualization for each terrain."""

  cfg: TerrainVelocityCommandCfg

  def __init__(self, cfg: TerrainVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
      raise ValueError("TerrainVelocityCommand requires generated terrain.")
    generator = terrain.cfg.terrain_generator
    if not generator.curriculum:
      raise ValueError("TerrainVelocityCommand requires curriculum=True for fixed terrain columns.")
    self.terrain_names = tuple(generator.sub_terrains)
    if set(self.terrain_names) != set(cfg.terrain_ranges):
      raise ValueError("terrain_ranges must exactly match terrain_generator.sub_terrains names.")
    # Keep mutable curriculum ranges separate from the declarative configuration.
    self.velocity_ranges = torch.tensor([
      [getattr(cfg.terrain_ranges[name], axis)
       for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]
      for name in self.terrain_names
    ], device=self.device)

  @property
  def terrain_ids(self) -> torch.Tensor:
    # MJLab 1.6 的课程地图按 sub_terrains 顺序为每类地形生成一列。
    return self._env.scene.terrain.terrain_types

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    terrain_ids = self.terrain_ids[env_ids]
    for terrain_id, name in enumerate(self.terrain_names):
      ids = env_ids[terrain_ids == terrain_id]
      if ids.numel() == 0:
        continue
      settings = self.cfg.terrain_ranges[name]
      bounds = self.velocity_ranges[terrain_id]
      self.vel_command_b[ids] = bounds[:, 0] + torch.rand(
        (len(ids), 3), device=self.device
      ) * (bounds[:, 1] - bounds[:, 0])
      self.vel_command_b[ids, :2] *= (
        self.vel_command_b[ids, :2].norm(dim=1) > settings.min_lin_speed
      ).unsqueeze(1)
      self.heading_target[ids] = torch.empty(len(ids), device=self.device).uniform_(
        *settings.heading
      )
      self.is_heading_env[ids] = (
        torch.rand(len(ids), device=self.device) < self.cfg.rel_heading_envs
      )
      # Zero translation while keeping heading control active.
      zero_ids = ids[torch.rand(len(ids), device=self.device) < settings.zero_command_probability]
      self.vel_command_b[zero_ids, :2] = 0.0
      self.is_standing_env[ids] = (
        torch.rand(len(ids), device=self.device) < self.cfg.rel_standing_envs
      )

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # The native implementation wraps heading error and applies standing masks.
    super()._update_command(env_ids)
    # 原生航向控制是当前状态的纯函数，会刷新全部环境；地形限幅同步覆盖全部行。
    bounds = self.velocity_ranges[self.terrain_ids, 2]
    self.vel_command_b[:, 2] = torch.clamp(
      self.vel_command_b[:, 2], min=bounds[:, 0], max=bounds[:, 1]
    )
    self.vel_command_b[self.is_standing_env] = 0.0


@dataclass(kw_only=True)
class TerrainVelocityCommandCfg(UniformVelocityCommandCfg):
  heading_command: bool = True
  heading_control_stiffness: float = 2.0
  ranges: UniformVelocityCommandCfg.Ranges = field(default_factory=lambda: UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5),
    ang_vel_z=(-math.pi, math.pi), heading=(-math.pi, math.pi),
  ))

  @dataclass(kw_only=True)
  class Ranges(UniformVelocityCommandCfg.Ranges):
    lin_vel_x_limits: tuple[float, float]
    lin_vel_y_limits: tuple[float, float]
    zero_command_probability: float = 0.05
    min_lin_speed: float = 0.0

  terrain_ranges: dict[str, Ranges]

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.init_velocity_prob != 0.0:
      raise ValueError("Terrain velocity commands require init_velocity_prob=0.")

  def build(self, env: ManagerBasedRlEnv) -> TerrainVelocityCommand:
    return TerrainVelocityCommand(self, env)
