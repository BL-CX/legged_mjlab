"""Read randomized simulator parameters independently from instantaneous state."""

import torch
from torch import Tensor


class SimContext:
    """Bind native robot indices for friction, encoder bias, and COM offset."""

    def __init__(self, env):
        self.env = env
        robot = env.scene["robot"]
        first_foot = robot.joint_names[0].split("_")[0]
        self.foot_id = env.sim.mj_model.geom(f"robot/{first_foot}_foot_collision").id
        self.base_id = env.sim.mj_model.body("robot/base_link").id
        self.nominal_com = torch.as_tensor(env.sim.mj_model.body_ipos[self.base_id].copy(), device=env.device)

    def read(self, ids) -> Tensor:
        """Read the parameters that affect dynamics and privileged observations."""
        env = self.env
        friction = env.sim.model.geom_friction
        friction = (
            friction[ids, self.foot_id, 0] if friction.ndim == 3 else friction[self.foot_id, 0].expand(ids.numel())
        )
        ipos = env.sim.model.body_ipos
        com = (
            ipos[ids, self.base_id] if ipos.ndim == 3 else ipos[self.base_id].expand(ids.numel(), -1)
        ) - self.nominal_com
        return torch.cat((friction[:, None], env.scene["robot"].data.encoder_bias[ids], com), -1).float()
