"""Go2 HIM and DreamWaQ training configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ...rsl_rl.config import (
    CTSActorCriticCfg,
    CTSAlgorithmCfg,
    CTSOnPolicyRunnerCfg,
    CTSAMPActorCriticCfg,
    CTSAMPAlgorithmCfg,
    CTSAMPOnPolicyRunnerCfg,

    DreamWaQActorCriticCfg,
    DreamWaQPPOAlgorithmCfg,
    DreamWaQOnPolicyRunnerCfg,
    DreamWaQAMPActorCriticCfg,
    DreamWaQAMPPPOAlgorithmCfg,
    DreamWaQAMPOnPolicyRunnerCfg,

    HIMAMPOnPolicyRunnerCfg,
    HIMAMPActorCriticCfg,
    HIMAMPPPOAlgorithmCfg,
    HIMActorCriticCfg,
    HIMOnPolicyRunnerCfg,
    HIMPPOAlgorithmCfg,
)


_GO2_MOTION_DIR = Path(__file__).resolve().parents[4] / "assets/motions/mocap_motions_go2"


def him_runner_cfg(terrain: Literal["rough", "flat"] = "rough") -> HIMOnPolicyRunnerCfg:
    """Go2 HIM settings from HimLoco-AMP-Lab's PPORunnerCfg.

    Environment dimensions are supplied by the environment adapter. Serialize
    this configuration with dataclasses.asdict for HIMOnPolicyRunner.
    """
    return HIMOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=50000,
        save_interval=100,
        experiment_name=f"go2_him_{terrain}",
        policy=HIMActorCriticCfg(
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            init_noise_std=1.0,
            activation="elu",
        ),
        algorithm=HIMPPOAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


def him_amp_runner_cfg(terrain: Literal["rough", "flat"] = "rough") -> HIMAMPOnPolicyRunnerCfg:
    """Explicit Go2 HIM-AMP training settings."""
    return HIMAMPOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=50000,
        save_interval=100,
        experiment_name=f"go2_him_amp_{terrain}",
        policy=HIMAMPActorCriticCfg(
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            init_noise_std=1.0,
            activation="elu",
        ),
        algorithm=HIMAMPPPOAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,

            # AMP settings
            amp_replay_buffer_size=1_000_000,
        ),

        # AMP settings
        amp_motion_files=[str(path) for path in sorted(_GO2_MOTION_DIR.glob("*.txt"))],
        reorder_motion_data=False,
        amp_reward_coef=0.15,
        amp_task_reward_lerp=0.9,
        amp_num_preload_transitions=200_000,
        amp_discr_hidden_dims=[1024, 512],
        min_normalized_std=[0.01] * 12,
    )


def dreamwaq_runner_cfg(terrain: Literal["rough", "flat"] = "rough") -> DreamWaQOnPolicyRunnerCfg:
    """Explicit Go2 DREAMWAQ training settings."""
    return DreamWaQOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=50000,
        save_interval=100,
        experiment_name=f"go2_dreamwaq_{terrain}",
        policy=DreamWaQActorCriticCfg(
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            init_noise_std=1.0,
            activation="elu",

            # vae settings
            num_latent_dims=16,
            num_explicit_dims=3,
        ),
        algorithm=DreamWaQPPOAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,

            # vae settings
            vae_learning_rate=1.0e-3,
            vae_kl_weight=1.0,

            # Symmetry loss settings
            sym_loss=False,
            sym_coef=1.0,
            frame_stack=5,
        ),
    )


def dreamwaq_amp_runner_cfg(terrain: Literal["rough", "flat"] = "rough") -> DreamWaQAMPOnPolicyRunnerCfg:
    """Explicit Go2 DREAMWAQ-AMP training settings."""
    return DreamWaQAMPOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=50000,
        save_interval=100,
        experiment_name=f"go2_dreamwaq_amp_{terrain}",
        policy=DreamWaQAMPActorCriticCfg(
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            init_noise_std=1.0,
            activation="elu",

            # vae settings
            num_latent_dims=16,
            num_explicit_dims=3,
        ),
        algorithm=DreamWaQAMPPPOAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,

            # vae settings
            vae_learning_rate=1.0e-3,
            vae_kl_weight=1.0,
            sym_loss=False,
            sym_coef=1.0,
            frame_stack=5,
            # AMP settings
            amp_replay_buffer_size=1_000_000,
        ),
        # AMP settings
        amp_motion_files=[str(path) for path in sorted(_GO2_MOTION_DIR.glob("*.txt"))],
        reorder_motion_data=False,
        amp_reward_coef=0.15,
        amp_task_reward_lerp=0.9,
        amp_num_preload_transitions=200_000,
        amp_discr_hidden_dims=[1024, 512],
        min_normalized_std=[0.01] * 12,
    )


def cts_runner_cfg(terrain: Literal["rough", "flat"] = "rough") -> CTSOnPolicyRunnerCfg:
    """Go2 CTS settings with separate teacher PPO and student distillation."""
    return CTSOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=20000,
        save_interval=500,
        experiment_name=f"go2_cts_{terrain}",
        policy=CTSActorCriticCfg(
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            teacher_encoder_hidden_dims=[512, 256],
            student_encoder_hidden_dims=[512, 256],
            init_noise_std=1.0,
            activation="elu",
            latent_dim=32,
            norm_type="l2norm",
        ),
        algorithm=CTSAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            student_encoder_learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            teacher_env_ratio=0.75,
            sym_loss=False,
            frame_stack=5,
        ),
    )


def cts_amp_runner_cfg(terrain: Literal["rough", "flat"] = "rough") -> CTSAMPOnPolicyRunnerCfg:
    """Go2 CTS settings with separate teacher PPO and student distillation."""
    return CTSAMPOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=20000,
        save_interval=500,
        experiment_name=f"go2_cts_amp_{terrain}",
        policy=CTSAMPActorCriticCfg(
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            teacher_encoder_hidden_dims=[512, 256],
            student_encoder_hidden_dims=[512, 256],
            init_noise_std=1.0,
            activation="elu",
            latent_dim=32,
            norm_type="l2norm",
        ),
        algorithm=CTSAMPAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            student_encoder_learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            teacher_env_ratio=0.75,
            sym_loss=False,
            frame_stack=5,
            amp_replay_buffer_size=1_000_000,
        ),
        amp_motion_files=[str(path) for path in sorted(_GO2_MOTION_DIR.glob("*.txt"))],
        reorder_motion_data=False,
        amp_reward_coef=0.15,
        amp_task_reward_lerp=0.9,
        amp_num_preload_transitions=2_000_000,
        amp_discr_hidden_dims=[1024, 512],
        min_normalized_std=[0.05] * 12,
    )
