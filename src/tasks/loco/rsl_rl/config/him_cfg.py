"""HIM and HIM-AMP initialization, following HimLoco-AMP-Lab's class layout.

Uses dataclasses and MJLab metadata instead of IsaacLab's configclass. Only
implemented parameters are exposed; observation normalization and AMP reward
interpolation are not supported by the current merged backend.

Example::

    from dataclasses import asdict

    cfg = HIMOnPolicyRunnerCfg(num_steps_per_env=24, max_iterations=10001)
    runner = HIMOnPolicyRunner(env, asdict(cfg), log_dir, device)

    cfg = HIMAMPOnPolicyRunnerCfg(amp_motion_files=["/path/to/motion.json"])
    runner = HIMAMPOnPolicyRunner(env, asdict(cfg), log_dir, device)

The wrapped env supplies num_obs, num_one_step_obs, num_privileged_obs and
num_actions. AMP also requires dt, dof_pos_limits and 30D AMP states.
"""

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


@dataclass
class HIMAMPActorCriticCfg(HIMActorCriticCfg):
    fixed_std: bool = False


@dataclass
class HIMAMPPPOAlgorithmCfg(HIMPPOAlgorithmCfg):
    amp_replay_buffer_size: int = 1_000_000


@dataclass
class HIMAMPOnPolicyRunnerCfg(HIMOnPolicyRunnerCfg):
    """Configure the merged 30D AMP backend with explicit motion paths.

    Provide motion paths when launching training. The motion loader reads the
    files and their contents.
    Replay capacity belongs only to algorithm; the runner derives min_std from
    min_normalized_std and action-ordered joint limits.
    """

    class_name: Literal["HIMAMPOnPolicyRunner"] = "HIMAMPOnPolicyRunner"
    policy_class_name: Literal["HIMAMPActorCritic"] = "HIMAMPActorCritic"
    algorithm_class_name: Literal["HIMAMPPPO"] = "HIMAMPPPO"
    policy: HIMAMPActorCriticCfg = field(default_factory=HIMAMPActorCriticCfg)
    algorithm: HIMAMPPPOAlgorithmCfg = field(default_factory=HIMAMPPPOAlgorithmCfg)
    amp_reward_coef: float = 0.2
    amp_motion_files: list[str] = field(default_factory=list)
    reorder_motion_data: bool = False
    amp_num_preload_transitions: int = 2_000_000
    amp_discr_hidden_dims: list[int] = field(default_factory=lambda: [1024, 512])
    min_normalized_std: float | list[float] = 0.01
