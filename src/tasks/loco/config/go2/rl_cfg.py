"""Go2 HIM and HIM-AMP training configurations."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

from ...rsl_rl.config import (
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
    """Use the HIM PPO settings with HimLoco-AMP-Lab's AMP defaults.

    Use the bundled Go2 mocap motions by default. Override them through
    --agent.amp-motion-files when launching training.
    """
    him = him_runner_cfg(terrain)
    return HIMAMPOnPolicyRunnerCfg(
        num_steps_per_env=him.num_steps_per_env,
        max_iterations=him.max_iterations,
        save_interval=him.save_interval,
        experiment_name=f"go2_him_amp_{terrain}",
        policy=HIMAMPActorCriticCfg(**asdict(him.policy)),
        algorithm=HIMAMPPPOAlgorithmCfg(**asdict(him.algorithm)),
        amp_motion_files=[str(path) for path in sorted(_GO2_MOTION_DIR.glob("*.txt"))],
        reorder_motion_data=False,
        amp_reward_coef=0.15,
        amp_num_preload_transitions=200_000,
        amp_discr_hidden_dims=[1024, 512],
        min_normalized_std=[0.01] * 12,
    )
