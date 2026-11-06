"""Load a blind policy and run it in the native velocity environment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.wrappers import VideoRecorder

from .export import checkpoint_actor, load_policy_checkpoint


def _restore_curriculum(command, checkpoint):
    saved = checkpoint["curriculum"]
    terrain = saved["terrain"]
    families, levels = terrain["family_ids"], terrain["levels"]
    rows = torch.arange(command.num_envs, device=families.device) % families.numel()
    command.load_curriculum_state_dict(
        {
            "command": saved["command"],
            "terrain": {
                **terrain,
                "family_ids": families[rows],
                "levels": levels[rows],
            },
        }
    )


def make_playback(
    checkpoint_file=None,
    *,
    agent="trained",
    env_cfg: ManagerBasedRlEnvCfg | None = None,
    num_envs: int | None = None,
    device="cpu",
    video=False,
    video_length=200,
):
    """Construct native playback with checkpoint semantics and optional video."""
    checkpoint = load_policy_checkpoint(checkpoint_file) if agent == "trained" else None
    if env_cfg is None:
        if checkpoint is None:
            raise ValueError("Untrained playback requires an environment configuration")
        env_cfg = checkpoint["config"].env
    cfg = deepcopy(env_cfg)
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    cfg.recorders.clear()
    cfg.terminations.pop("time_out", None)
    task_cfg = cfg.commands["velocity"]
    task_cfg.play = True
    task_cfg.terrain_curriculum = False
    task_cfg.noise.add_noise = False
    path = Path(checkpoint_file).expanduser().resolve() if checkpoint_file is not None else None
    record_video = path is not None and video
    torch.manual_seed(cfg.seed)
    env = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array" if record_video else None)
    try:
        command = env.command_manager.get_term("velocity")
        model = None
        if checkpoint is not None:
            model = checkpoint_actor(checkpoint).to(env.device)
            _restore_curriculum(command, checkpoint)
        env.reset()

        @torch.no_grad()
        def policy(observations):
            if model is not None:
                return model(observations["history"])
            shape = (env.num_envs, command.model.spec.act_dim)
            if agent == "zero":
                return torch.zeros(shape, device=env.device)
            return 2 * torch.rand(shape, device=env.device) - 1

        if record_video:
            env = VideoRecorder(
                env,
                video_folder=path.parent / "videos" / "play",
                step_trigger=lambda step: step == 0,
                video_length=video_length,
                disable_logger=True,
            )
        return env, policy
    except BaseException:
        env.close()
        raise
