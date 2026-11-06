# Copyright (c) 2026-2027 zh
"""Hard fall detection and differentiable termination probabilities."""

import math

import torch
from torch import Tensor

from ...world_model.state_spec import Physical, TaskState


def hard_termination(state: TaskState, fall_angle: float) -> Tensor:
    return Physical.split(state.physical).gravity[:, 2] > -math.cos(fall_angle)


def soft_termination(state: TaskState, fall_angle: float, fall_softness_deg: float) -> Tensor:
    tilt = torch.acos((-Physical.split(state.physical).gravity[:, 2]).clamp(-1 + 1e-4, 1 - 1e-4))
    probability = torch.sigmoid((tilt - fall_angle) / math.radians(fall_softness_deg))
    return probability
