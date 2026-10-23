"""DreamWaQ configuration for the shared six-frame HIM environment interface."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class DreamWaQActorCriticCfg:
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    init_noise_std: float = 1.0
    num_latent_dims: int = 16
    num_explicit_dims: int = 3


@dataclass
class DreamWaQPPOAlgorithmCfg:
    num_learning_epochs: int = 1
    num_mini_batches: int = 1
    clip_param: float = 0.2
    gamma: float = 0.998
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: Literal["fixed", "adaptive"] = "fixed"
    desired_kl: float = 0.01
    vae_learning_rate: float = 1e-3
    vae_kl_weight: float = 1.0
    sym_loss: bool = False
    sym_coef: float = 1.0
    frame_stack: int = 5
    obs_permutation: list[int] | None = None
    act_permutation: list[int] | None = None


@dataclass
class DreamWaQOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    seed: int = 1
    logger: Literal["tensorboard"] = "tensorboard"
    upload_model: bool = False
    class_name: Literal["DreamWaQOnPolicyRunner"] = "DreamWaQOnPolicyRunner"
    policy_class_name: Literal["DreamWaQActorCritic"] = "DreamWaQActorCritic"
    algorithm_class_name: Literal["DreamWaQPPO"] = "DreamWaQPPO"
    policy: DreamWaQActorCriticCfg = field(default_factory=DreamWaQActorCriticCfg)
    algorithm: DreamWaQPPOAlgorithmCfg = field(default_factory=DreamWaQPPOAlgorithmCfg)


@dataclass
class DreamWaQAMPActorCriticCfg(DreamWaQActorCriticCfg):
    """AMP uses the same VAE actor-critic; no unused policy flags are exposed."""


@dataclass
class DreamWaQAMPPPOAlgorithmCfg(DreamWaQPPOAlgorithmCfg):
    amp_replay_buffer_size: int = 1_000_000


@dataclass
class DreamWaQAMPOnPolicyRunnerCfg(DreamWaQOnPolicyRunnerCfg):
    class_name: Literal["DreamWaQAMPOnPolicyRunner"] = "DreamWaQAMPOnPolicyRunner"
    algorithm_class_name: Literal["DreamWaQAMPPPO"] = "DreamWaQAMPPPO"
    policy: DreamWaQAMPActorCriticCfg = field(default_factory=DreamWaQAMPActorCriticCfg)
    algorithm: DreamWaQAMPPPOAlgorithmCfg = field(default_factory=DreamWaQAMPPPOAlgorithmCfg)
    amp_reward_coef: float = 0.2
    amp_motion_files: list[str] = field(default_factory=list)
    reorder_motion_data: bool = False
    amp_num_preload_transitions: int = 2_000_000
    amp_discr_hidden_dims: list[int] = field(default_factory=lambda: [1024, 512])
    min_normalized_std: float | list[float] = 0.01
