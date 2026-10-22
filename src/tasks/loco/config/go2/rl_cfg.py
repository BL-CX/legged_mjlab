"""Go2 HIM and HIM-AMP training configurations."""

from __future__ import annotations

from typing import Literal

from ...rsl_rl.config import (
    HIMActorCriticCfg,
    HIMOnPolicyRunnerCfg,
    HIMPPOAlgorithmCfg,
)




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
