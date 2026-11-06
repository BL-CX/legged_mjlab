"""Play a registered SuperMimic task with its native environment configuration."""

# ruff: noqa: E402 -- support direct execution before importing repository packages.

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.supermimic.eval.playback import make_playback
from src.tasks.supermimic.training.config import SuperMimicCfg


@dataclass(frozen=True)
class PlayConfig:
    """Playback settings; use steps for a finite headless run."""

    env: ManagerBasedRlEnvCfg
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: Path | None = None
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    steps: int | None = None
    video: bool = False
    video_length: int = 200


def run_play(cfg: PlayConfig):
    if cfg.agent == "trained" and cfg.checkpoint_file is None:
        raise ValueError("Trained playback requires --checkpoint-file")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env, policy = make_playback(
        env_cfg=cfg.env,
        checkpoint_file=cfg.checkpoint_file,
        agent=cfg.agent,
        device=device,
        video=cfg.video,
        video_length=cfg.video_length,
    )
    try:
        if cfg.steps is not None:
            observations = env.get_observations()
            with torch.inference_mode():
                for _ in range(cfg.steps):
                    observations, *_ = env.step(policy(observations))
            print(f"[INFO] Completed {cfg.steps} playback steps on {device}")
            return
        viewer = cfg.viewer
        if viewer == "auto":
            viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
        from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

        viewer_cls = NativeMujocoViewer if viewer == "native" else ViserPlayViewer
        viewer_cls(env, policy).run()
    finally:
        env.close()


def main():
    cfg = tyro.extras.overridable_config_cli(
        {
            task_id: (f"Play {task_id}.", PlayConfig(env=load_env_cfg(task_id, play=True)))
            for task_id in list_tasks()
            if isinstance(load_rl_cfg(task_id), SuperMimicCfg)
        },
        config=mjlab.TYRO_FLAGS,
    )
    run_play(cfg)


if __name__ == "__main__":
    main()
