"""Load a blind policy and run it in the native velocity environment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.wrappers import VideoRecorder

from ..config.go2.env_cfg import flat_env_cfg, rough_env_cfg
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
    terrain=None,
    num_envs=1,
    device="cpu",
    video=False,
    video_length=200,
    video_height=None,
    video_width=None,
    no_terminations=False,
):
    """Construct native playback with checkpoint semantics and optional video."""
    checkpoint = load_policy_checkpoint(checkpoint_file) if agent == "trained" else None
    if checkpoint is not None:
        cfg = deepcopy(checkpoint["config"].env)
        saved_terrain = "flat" if cfg.scene.terrain.terrain_type == "plane" else "rough"
        if terrain is not None and terrain != saved_terrain:
            raise ValueError("Playback terrain differs from the saved configuration")
    else:
        cfg = rough_env_cfg(play=True) if terrain == "rough" else flat_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.recorders.clear()
    cfg.terminations.pop("time_out", None)
    task_cfg = cfg.commands["velocity"]
    task_cfg.play = True
    task_cfg.terrain_curriculum = False
    task_cfg.no_terminations = no_terminations
    task_cfg.noise.add_noise = False
    if video_height is not None:
        cfg.viewer.height = video_height
    if video_width is not None:
        cfg.viewer.width = video_width
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
