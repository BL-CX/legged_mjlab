"""CTS teacher/student training with MJLab's shared locomotion interface."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class CTSActorCriticCfg:
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    teacher_encoder_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    student_encoder_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    activation: str = "elu"
    init_noise_std: float = 1.0
    latent_dim: int = 32
    norm_type: Literal["l2norm", "simnorm"] = "l2norm"


@dataclass
class CTSAlgorithmCfg:
    num_learning_epochs: int = 1
    num_mini_batches: int = 1
    clip_param: float = 0.2
    gamma: float = 0.998
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    learning_rate: float = 1e-3
    student_encoder_learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: Literal["fixed", "adaptive"] = "fixed"
    desired_kl: float = 0.01
    teacher_env_ratio: float = 0.75
    sym_loss: bool = False
    sym_coef: float = 1.0
    frame_stack: int = 5
    obs_permutation: list[float] | None = None
    act_permutation: list[float] | None = None
    privileged_permutation: list[float] | None = None


@dataclass
class CTSOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    seed: int = 1
    logger: Literal["tensorboard"] = "tensorboard"
    upload_model: bool = False
    class_name: Literal["CTSOnPolicyRunner"] = "CTSOnPolicyRunner"
    policy_class_name: Literal["CTSActorCritic"] = "CTSActorCritic"
    algorithm_class_name: Literal["CTS"] = "CTS"
    policy: CTSActorCriticCfg = field(default_factory=CTSActorCriticCfg)
    algorithm: CTSAlgorithmCfg = field(default_factory=CTSAlgorithmCfg)


@dataclass
class CTSAMPActorCriticCfg(CTSActorCriticCfg):
    """AMP shares the same teacher and student encoders."""


@dataclass
class CTSAMPAlgorithmCfg(CTSAlgorithmCfg):
    amp_replay_buffer_size: int = 1_000_000


@dataclass
class CTSAMPOnPolicyRunnerCfg(CTSOnPolicyRunnerCfg):
    class_name: Literal["CTSAMPOnPolicyRunner"] = "CTSAMPOnPolicyRunner"
    algorithm_class_name: Literal["CTSAMP"] = "CTSAMP"
    policy: CTSAMPActorCriticCfg = field(default_factory=CTSAMPActorCriticCfg)
    algorithm: CTSAMPAlgorithmCfg = field(default_factory=CTSAMPAlgorithmCfg)
    amp_reward_coef: float = 0.15
    amp_task_reward_lerp: float = 0.9
    amp_motion_files: list[str] = field(default_factory=list)
    reorder_motion_data: bool = False
    amp_num_preload_transitions: int = 2_000_000
    amp_discr_hidden_dims: list[int] = field(default_factory=lambda: [1024, 512])
    min_normalized_std: float | list[float] = 0.01
