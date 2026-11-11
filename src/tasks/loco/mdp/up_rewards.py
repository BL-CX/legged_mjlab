from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")




def lin_vel_z_l2_up(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize vertical base velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity_b = asset.data.projected_gravity_b
    return asset.data.root_link_lin_vel_b[:, 2].square()*torch.clamp(-projected_gravity_b[:, 2], 0, 1)


def ang_vel_xy_l2_up(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize roll and pitch angular velocity."""
    asset: Entity = env.scene[asset_cfg.name]

    projected_gravity_b = asset.data.projected_gravity_b
    return asset.data.root_link_ang_vel_b[:, :2].square().sum(dim=1)*torch.clamp(-projected_gravity_b[:, 2], 0, 1)


def body_orientation_l2_up(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize roll and pitch of the selected body."""
    asset: Entity = env.scene[asset_cfg.name]

    projected_gravity_b = asset.data.projected_gravity_b
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)*torch.clamp(-projected_gravity_b[:, 2], 0, 1)


def base_height_reward_up(
    env: ManagerBasedRlEnv,
    target_height: float,
    sensor_name: str | None = None,
    base_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[base_cfg.name]
  base_z = asset.data.root_link_pos_w[:, 2]
  projected_gravity_b = asset.data.projected_gravity_b

  if base_cfg.body_ids:
    base_z = asset.data.body_link_pos_w[:, base_cfg.body_ids, 2].squeeze(1)

  if sensor_name is None:
    measured_height = base_z
  else:
    sensor = env.scene[sensor_name]
    hit_pos_w = getattr(sensor.data, "hit_pos_w", None)
    if hit_pos_w is None:
      raise TypeError(f"Sensor '{sensor_name}' must provide ray hit positions.")
    measured_height = base_z - hit_pos_w[..., 2].mean(dim=1)
  return torch.square(measured_height - target_height)*torch.clamp(-projected_gravity_b[:, 2], 0, 1)


def feet_clearance_up(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]

  projected_gravity_b = asset.data.projected_gravity_b

  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids] - asset.data.root_link_pos_w[:, None, :]
  foot_vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids] - asset.data.root_link_lin_vel_w[:, None, :]
  num_feet = foot_pos.shape[1]
  # mjlab's TorchScript quaternion helper requires matching leading shapes;
  # it does not broadcast [B, 1, 4] against [B, N, 3].
  root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, num_feet, -1)
  foot_pos_b = quat_apply_inverse(root_quat, foot_pos)
  foot_vel_b = quat_apply_inverse(root_quat, foot_vel)
  height_error = torch.square(foot_pos_b[:, :, 2] - target_height)
  lateral_speed = torch.linalg.vector_norm(foot_vel_b[:, :, :2], dim=-1)
  cost = torch.sum(height_error * lateral_speed, dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost*torch.clamp(-projected_gravity_b[:, 2], 0, 1)


def upward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize roll and pitch of the selected body."""
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity_b = asset.data.projected_gravity_b
    return 1 - projected_gravity_b[:, 2]


def feet_slip_up(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_b = asset.data.projected_gravity_b

  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost*torch.clamp(-projected_gravity_b[:, 2], 0, 1)


def feet_stumble_up(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize feet hitting vertical surfaces using world-frame contact forces."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_b = asset.data.projected_gravity_b

  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.force is not None
  forces = contact_sensor.data.force  # [B, N, 3], world-frame foot forces
  horizontal_force = torch.norm(forces[:, :, :2], dim=2)  # [B, N]
  vertical_force = torch.abs(forces[:, :, 2])  # [B, N]
  cost = torch.any(horizontal_force > 5.0 * vertical_force, dim=1).float() * active
  return cost * torch.clamp(-projected_gravity_b[:, 2], 0, 1)


def collision_up(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Count selected bodies with contact force above threshold, gated by uprightness.

  Use a netforce contact sensor with one primary per selected body.
  """
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_b = asset.data.projected_gravity_b

  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  forces = contact_sensor.data.force  # [B, N, 3], one net force per selected body
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  cost = torch.sum((force_magnitude > force_threshold).float(), dim=1)
  return cost * torch.clamp(-projected_gravity_b[:, 2], 0, 1)


def foot_mirror_up(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize diagonal-leg joint-angle differences, gated by uprightness.

  Selected joints must be ordered FL, FR, RL, RR, with hip, thigh, calf per leg.
  """
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_b = asset.data.projected_gravity_b

  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  assert joint_pos.shape[1] == 12, "foot_mirror_up requires 12 ordered leg joints."
  diff1 = torch.sum(torch.square(joint_pos[:, 0:3] - joint_pos[:, 9:12]), dim=-1)
  diff2 = torch.sum(torch.square(joint_pos[:, 3:6] - joint_pos[:, 6:9]), dim=-1)
  return 0.5 * torch.clamp(-projected_gravity_b[:, 2], 0, 1) * (diff1 + diff2)


def stand_still_up(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity_b = asset.data.projected_gravity_b
     
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward * (1 - projected_gravity_b[:, 2])
