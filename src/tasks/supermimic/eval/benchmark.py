# Copyright (c) 2026-2027 zh
"""Command-wise tracking and first-episode survival evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .playback import make_playback


@torch.no_grad()
def evaluate_policy(checkpoint: str | Path, *, zero_action=False, device="cpu", num_envs=32, seconds=20.0) -> dict:
    """Evaluate four command groups with shared checkpoint settings and initial seeds."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    env, actor = make_playback(checkpoint, num_envs=num_envs, device=device)
    try:
        command = env.command_manager.get_term("velocity")
        task = command.model
        cfg = command.task_cfg
        device = env.device
        commands = torch.tensor(((0.5, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)), device=device)
        command_names = ("forward", "backward", "lateral", "yaw")
        groups = torch.arange(num_envs, device=device) % 4
        fixed_commands = commands[groups]
        command.set_commands(fixed_commands)
        # Reset the initial physics state after all components are ready.
        torch.manual_seed(cfg.env.seed)
        observations, _ = env.reset()
        count = math.ceil(seconds / env.step_dt)
        first_steps = torch.full((num_envs,), count, device=device, dtype=torch.long)
        first_done = torch.zeros(num_envs, device=device, dtype=torch.bool)
        first_failed = first_done.clone()
        velocity_sum = torch.zeros(num_envs, 3, device=device, dtype=torch.float64)
        linear_square, yaw_square, action_square, height_sum, reward_sum = (
            torch.zeros(num_envs, device=device, dtype=torch.float64) for _ in range(5)
        )
        failure_counts = torch.zeros(num_envs, device=device, dtype=torch.long)
        timeout_counts = failure_counts.clone()
        action_max = 0.0
        for step in range(count):
            if zero_action:
                actions = torch.zeros(num_envs, task.spec.act_dim, device=device)
            else:
                actions = actor(observations)
            observations, reward, terminated, truncated, info = env.step(actions)
            final = command.final_state
            measured = task.tracking_velocity(final)
            velocity_sum += measured.double()
            linear_square += (measured[:, :2] - fixed_commands[:, :2]).double().square().sum(-1)
            yaw_square += (measured[:, 2] - fixed_commands[:, 2]).double().square()
            action_square += actions.double().square().mean(-1)
            height_sum += task.spec.physical_features(final)[:, 0].double()
            reward_sum += reward.double()
            action_max = max(action_max, float(actions.abs().max()))
            done = terminated | truncated
            ended = done & ~first_done
            first_steps[ended] = step + 1
            first_failed[ended] = terminated[ended]
            first_done |= done
            failure_counts += terminated.long()
            timeout_counts += (truncated & ~terminated).long()

        def summarize(rows):
            mean_velocity = velocity_sum[rows].mean(0) / count
            return {
                "num_envs": int(rows.sum()),
                "mean_vx_mps": float(mean_velocity[0]),
                "mean_vy_mps": float(mean_velocity[1]),
                "mean_yaw_rate_radps": float(mean_velocity[2]),
                "linear_velocity_rmse_mps": math.sqrt(float(linear_square[rows].mean()) / count),
                "yaw_rate_rmse_radps": math.sqrt(float(yaw_square[rows].mean()) / count),
                "first_episode_mean_seconds": float(first_steps[rows].double().mean()) * env.step_dt,
                "first_episode_failure_fraction": float(first_failed[rows].double().mean()),
                "first_episode_censored_fraction": float((~first_done[rows]).double().mean()),
                "failures": int(failure_counts[rows].sum()),
                "timeouts": int(timeout_counts[rows].sum()),
                "mean_reward_per_step": float(reward_sum[rows].mean()) / count,
                "mean_return_per_environment": float(reward_sum[rows].mean()),
                "mean_base_height_m": float(height_sum[rows].mean()) / count,
                "action_rms": math.sqrt(float(action_square[rows].mean()) / count),
            }

        result = {
            "checkpoint": str(checkpoint),
            "terrain": cfg.env.terrain,
            "device": str(device),
            "seed": cfg.env.seed,
            "control_steps": count,
            "commands": commands.tolist(),
            "command_frame": "body_xy_world_yaw",
            "duration_seconds": count * env.step_dt,
            "zero_action": zero_action,
            **summarize(torch.ones(num_envs, device=device, dtype=torch.bool)),
            "per_command": {
                name: {"command": commands[index].tolist(), **summarize(groups == index)}
                for index, name in enumerate(command_names)
            },
            "falls_per_environment_minute": int(failure_counts.sum()) / (num_envs * count * env.step_dt / 60),
            "action_abs_max": action_max,
            "command_rms": float(fixed_commands.square().sum(-1).mean().sqrt()),
        }
        return result
    finally:
        env.close()
