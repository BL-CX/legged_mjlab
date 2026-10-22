from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def amp_observations(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Raw AMP state: joint positions, body linear/angular velocity, joint velocity.

    Go2 produces 30 values (12 + 3 + 3 + 12), matching AMPMotionLoader's preloaded
    expert states. Joint positions are absolute; no policy scales or offsets
    are applied. Joint selection/order follows asset_cfg for both joint fields.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.cat(
        (
            asset.data.joint_pos[:, asset_cfg.joint_ids],
            asset.data.root_link_lin_vel_b,
            asset.data.root_link_ang_vel_b,
            asset.data.joint_vel[:, asset_cfg.joint_ids],
        ),
        dim=-1,
    )


def base_external_force(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_external_force[:, asset_cfg.body_ids].squeeze(1)

def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
