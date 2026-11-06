# Copyright (c) 2026-2027 zh
"""Heading-frame vector rotations and planar pose integration."""

import torch
from torch import Tensor


def yaw_rotate_inverse(vector: Tensor, yaw: Tensor) -> Tensor:
    """Rotate [B, 3] or [B, K, 3] vectors inversely about world Z."""
    cosine, sine = yaw.cos(), yaw.sin()
    return torch.stack(
        (
            cosine * vector[..., 0] + sine * vector[..., 1],
            -sine * vector[..., 0] + cosine * vector[..., 1],
            vector[..., 2],
        ),
        -1,
    )


def pose_delta(before: Tensor, after: Tensor) -> Tensor:
    """Express translation in the starting heading frame and wrap the yaw delta to [-pi, pi]."""
    difference = after - before
    yaw = before[..., 2]
    return torch.stack(
        (
            yaw.cos() * difference[..., 0] + yaw.sin() * difference[..., 1],
            -yaw.sin() * difference[..., 0] + yaw.cos() * difference[..., 1],
            torch.atan2(difference[..., 2].sin(), difference[..., 2].cos()),
        ),
        -1,
    )


def apply_pose_delta(pose: Tensor, delta: Tensor) -> Tensor:
    yaw = pose[..., 2]
    return torch.stack(
        (
            pose[..., 0] + yaw.cos() * delta[..., 0] - yaw.sin() * delta[..., 1],
            pose[..., 1] + yaw.sin() * delta[..., 0] + yaw.cos() * delta[..., 1],
            yaw + delta[..., 2],
        ),
        -1,
    )
