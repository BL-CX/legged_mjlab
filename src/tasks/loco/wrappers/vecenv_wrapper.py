"""Local runner metadata on top of MJLab's standard environment interface."""

import torch
from mjlab.rl import RslRlVecEnvWrapper


class LocoVecEnvWrapper(RslRlVecEnvWrapper):
    @property
    def dof_pos_limits(self) -> torch.Tensor:
        """Soft joint limits [num_actions, 2], in action-manager order.

        Resolve each action term through the same actuator-to-joint mapping
        used by MJLab joint actions, including subsets and multiple terms.
        """
        limits = []
        for name in self.unwrapped.action_manager.active_terms:
            action = self.unwrapped.action_manager.get_term(name)
            entity = self.unwrapped.scene[action.cfg.entity_name]
            joint_ids, _ = entity.find_joints_by_actuator_names(action.cfg.actuator_names)
            limits.append(entity.data.soft_joint_pos_limits[0, joint_ids])
        joint_limits = torch.cat(limits)
        if joint_limits.shape != (self.num_actions, 2):
            raise ValueError("Joint limits must match the action dimension and order")
        return joint_limits
