"""Configuration dataclasses for PIE training."""

from dataclasses import dataclass, field

from mjlab.rl import RslRlPpoAlgorithmCfg


@dataclass
class PIEPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    auxiliary_loss_coef: float = 1.0
    velocity_loss_coef: float = 1.0
    foot_clearance_loss_coef: float = 1.0
    height_reconstruction_loss_coef: float = 1.0
    successor_loss_coef: float = 1.0
    kl_loss_coef: float = 4.0


@dataclass
class PIEAMPPpoAlgorithmCfg(PIEPpoAlgorithmCfg):
    """PIE training settings with an independent AMP discriminator."""

    class_name: str = "src.tasks.pie.rl.amp_ppo:PIEAMPPPO"
    amp_motion_files: list[str] = field(default_factory=list)
    reorder_motion_data: bool = False
    amp_reward_coef: float = 0.5
    amp_num_preload_transitions: int = 200_000
    amp_discr_hidden_dims: tuple[int, ...] = (1024, 512)
    amp_replay_buffer_size: int = 1_000_000
    amp_learning_rate: float = 1.0e-4
    amp_num_updates: int = 4
    amp_mini_batch_size: int = 1024
    amp_grad_penalty_coef: float = 10.0
    amp_max_grad_norm: float = 1.0
