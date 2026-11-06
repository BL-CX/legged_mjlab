# Copyright (c) 2026-2027 zh
"""Hard fall detection and differentiable termination probabilities."""

import math

import torch
from torch import Tensor

from ...world_model.state_spec import GRAVITY, TaskState


def hard_termination(state: TaskState, fall_angle: float) -> Tensor:
    return state.physical[:, GRAVITY.stop - 1] > -math.cos(fall_angle)


def soft_termination(state: TaskState, fall_angle: float, fall_softness_deg: float) -> Tensor:
    tilt = torch.acos((-state.physical[:, GRAVITY.stop - 1]).clamp(-1 + 1e-4, 1 - 1e-4))
    probability = torch.sigmoid((tilt - fall_angle) / math.radians(fall_softness_deg))
    return probability
