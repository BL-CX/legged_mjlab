"""Load a blind policy and run it in the native velocity environment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from mjlab.utils.wrappers import VideoRecorder

from ..config.go2.env_cfg import EnvCfg, make_env
from ..world_model.heads import AnalyticHeads
from .export import checkpoint_actor, load_policy_checkpoint


def _restore_curriculum(command, checkpoint):
    saved = checkpoint["curriculum"]
    terrain = saved["terrain"]
    families, levels = terrain["family_ids"], terrain["levels"]
    rows = torch.arange(command.num_envs, device=families.device) % families.numel()
    command.load_curriculum_state_dict(
        {
            "command": saved["command"],
            "terrain": {**terrain, "family_ids": families[rows], "levels": levels[rows]},
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
    source = load_policy_checkpoint(checkpoint_file) if agent == "trained" else None
    cfg = deepcopy(source[0]) if source is not None else EnvCfg()
    if terrain is not None:
        if source is not None and terrain != cfg.terrain:
            raise ValueError("Playback terrain differs from the saved configuration")
        cfg.terrain = terrain
    cfg.num_envs = num_envs
    path = Path(checkpoint_file).expanduser().resolve() if checkpoint_file is not None else None
    record_video = path is not None and video
    torch.manual_seed(cfg.seed)
    env = make_env(
        cfg,
        device=device,
        play=True,
        render_mode="rgb_array" if record_video else None,
        video_height=video_height,
        video_width=video_width,
        no_terminations=no_terminations,
    )
    try:
        command = env.command_manager.get_term("velocity")
        model = None
        if source is not None:
            saved_cfg, snapshot, checkpoint = source
            if snapshot["geometry_hash"] != command.surface.geometry_hash:
                raise ValueError("Playback terrain geometry differs from the saved task")
            original = AnalyticHeads(saved_cfg, env.scene["robot"], command.surface, device=env.device)
            if checkpoint["spec_hash"] != original.hash:
                raise ValueError("Checkpoint task spec does not match the saved configuration and current robot")
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
