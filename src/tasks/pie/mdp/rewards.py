"""Velocity tracking, body stability and terrain contact rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking of body-frame planar velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = (command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]).square().sum(dim=1)
  return torch.exp(-error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking of body-frame yaw velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = (command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]).square()
  return torch.exp(-error / std**2)


def lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize squared vertical body velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, 2].square()


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize squared roll and pitch angular velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b[:, :2].square().sum(dim=1)


def collision_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.1,
) -> torch.Tensor:
  """Count contact channels exceeding the force threshold.

  Use one channel per selected non-foot body to count colliding bodies.
  A channel is counted once even if several history samples exceed the threshold.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force = data.force_history.norm(dim=-1)
    return (force > force_threshold).any(dim=-1).sum(dim=1).float()
  if data.force is not None:
    return (data.force.norm(dim=-1) > force_threshold).sum(dim=1).float()
  assert data.found is not None
  return (data.found > 0).sum(dim=1).float()
