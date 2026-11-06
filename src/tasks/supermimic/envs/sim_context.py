# Copyright (c) 2026-2027 zh
"""Read randomized physics and world origins as external task context."""

import torch
from torch import Tensor

from ..world_model.state_spec import (
    CONTEXT_COM,
    CONTEXT_DIM,
    CONTEXT_ENCODER_BIAS,
    CONTEXT_FRICTION,
    CONTEXT_ORIGIN,
    StateSpec,
)


class SimContext:
    """Bind simulator parameters independently from instantaneous physical state."""

    def __init__(self, env, spec: StateSpec):
        self.env = env
        self.foot_id = env.sim.mj_model.geom(f"robot/{spec.foot_names[0]}_foot_collision").id
        self.base_id = env.sim.mj_model.body("robot/base_link").id
        self.nominal_com = torch.as_tensor(env.sim.mj_model.body_ipos[self.base_id].copy(), device=env.device)

    def read(self, ids) -> Tensor:
        """Read friction, COM offset, encoder bias, and the current terrain origin."""
        env = self.env
        context = torch.zeros((ids.numel(), CONTEXT_DIM), dtype=torch.float32, device=env.device)
        friction = env.sim.model.geom_friction
        context[:, CONTEXT_FRICTION] = (
            friction[ids, self.foot_id, 0] if friction.ndim == 3 else friction[self.foot_id, 0]
        )
        ipos = env.sim.model.body_ipos
        context[:, CONTEXT_COM] = (
            ipos[ids, self.base_id] if ipos.ndim == 3 else ipos[self.base_id]
        ) - self.nominal_com
        context[:, CONTEXT_ENCODER_BIAS] = env.scene["robot"].data.encoder_bias[ids]
        context[:, CONTEXT_ORIGIN] = env.scene.env_origins[ids]
        return context
