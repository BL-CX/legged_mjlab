"""Useful methods for MDP events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def reset_joints_by_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]

    default_pos = asset.data.default_joint_pos
    default_vel = asset.data.default_joint_vel
    soft_limits = asset.data.soft_joint_pos_limits

    assert default_pos is not None
    assert default_vel is not None
    assert soft_limits is not None

    joint_pos = default_pos[env_ids][:, asset_cfg.joint_ids].clone()
    scale = sample_uniform(
        position_range[0],
        position_range[1],
        joint_pos.shape,
        env.device,
    )
    joint_pos *= scale

    limits = soft_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos.clamp_(limits[..., 0], limits[..., 1])

    joint_vel = default_vel[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel += sample_uniform(
        velocity_range[0],
        velocity_range[1],
        joint_vel.shape,
        env.device,
    )

    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
        joint_ids=asset_cfg.joint_ids,
    )
