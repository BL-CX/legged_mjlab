"""Native terrain levels and per-terrain velocity curriculum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers import ManagerTermBase
from mjlab.tasks.velocity.mdp.curriculums import (
  commands_vel,
  terrain_levels_vel,
)

from .velocity_command import TerrainVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "commands_vel", "terrain_levels_vel", "TerrainVelocityCurriculum",
]


class TerrainVelocityCurriculum(ManagerTermBase):
  """Expand velocity ranges using independent windows for each terrain.

  Reward sums are already scaled by dt in MJLab, so normalization uses maximum
  episode seconds times reward weight.
  Early failures consequently reduce the score.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self.command_name = cfg.params.get("command_name", "twist")
    self.command = env.command_manager.get_term(self.command_name)
    if not isinstance(self.command, TerrainVelocityCommand):
      raise TypeError("TerrainVelocityCurriculum requires TerrainVelocityCommand.")
    count = len(self.command.terrain_names)
    self.reward_sums = torch.zeros(count, device=self.device)
    self.episode_counts = torch.zeros(count, dtype=torch.long, device=self.device)
    self.scores = torch.zeros(count, device=self.device)
    self.window_sizes = torch.bincount(self.command.terrain_ids, minlength=count)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str = "twist",
    reward_term_name: str = "track_linear_velocity",
    threshold: float = 0.8,
    increment: float = 0.2,
  ) -> dict[str, torch.Tensor]:
    if command_name != self.command_name:
      raise ValueError("The curriculum command_name must not change after initialization.")
    if not 0.0 <= threshold <= 1.0:
      raise ValueError("threshold must be in [0, 1].")
    weight = env.reward_manager.get_term_cfg(reward_term_name).weight
    if weight <= 0.0:
      raise ValueError("Command curriculum requires a positive tracking reward weight.")
    # CurriculumManager runs before reward reset; skip the initial reset.
    ids = env_ids[env.episode_length_buf[env_ids] > 0]
    terrain_ids = self.command.terrain_ids[ids]
    stats = {}
    for terrain_id, name in enumerate(self.command.terrain_names):
      selected = ids[terrain_ids == terrain_id]
      self.reward_sums[terrain_id] += env.reward_manager._episode_sums[reward_term_name][selected].sum()
      self.episode_counts[terrain_id] += selected.numel()
      if self.window_sizes[terrain_id] > 0 and self.episode_counts[terrain_id] >= self.window_sizes[terrain_id]:
        score = self.reward_sums[terrain_id] / (
          self.episode_counts[terrain_id] * env.max_episode_length_s * weight
        )
        self.scores[terrain_id] = score
        if score > threshold:
          settings = self.command.cfg.terrain_ranges[name]
          for axis, limits in enumerate((settings.lin_vel_x_limits, settings.lin_vel_y_limits)):
            bounds = self.command.velocity_ranges[terrain_id, axis]
            bounds[0] = torch.clamp(bounds[0] - increment, min=limits[0])
            bounds[1] = torch.clamp(bounds[1] + increment, max=limits[1])
        self.reward_sums[terrain_id] = 0.0
        self.episode_counts[terrain_id] = 0
      stats[f"{name}/tracking_score"] = self.scores[terrain_id].clone()
      for axis, axis_name in enumerate(("lin_vel_x", "lin_vel_y")):
        bounds = self.command.velocity_ranges[terrain_id, axis]
        stats[f"{name}/{axis_name}_min"] = bounds[0].clone()
        stats[f"{name}/{axis_name}_max"] = bounds[1].clone()
    return stats
