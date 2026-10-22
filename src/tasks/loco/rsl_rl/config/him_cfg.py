"""HIM learning configuration for MJLab."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class HIMActorCriticCfg:
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    init_noise_std: float = 1.0


@dataclass
class HIMPPOAlgorithmCfg:
    num_learning_epochs: int = 1
    num_mini_batches: int = 1
    clip_param: float = 0.2
    gamma: float = 0.998
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: Literal["fixed", "adaptive"] = "fixed"
    desired_kl: float = 0.01


@dataclass
class HIMOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    seed: int = 1
    logger: Literal["tensorboard"] = "tensorboard"
    upload_model: bool = False
    class_name: Literal["HIMOnPolicyRunner"] = "HIMOnPolicyRunner"
    policy_class_name: Literal["HIMActorCritic"] = "HIMActorCritic"
    algorithm_class_name: Literal["HIMPPO"] = "HIMPPO"
    policy: HIMActorCriticCfg = field(default_factory=HIMActorCriticCfg)
    algorithm: HIMPPOAlgorithmCfg = field(default_factory=HIMPPOAlgorithmCfg)
