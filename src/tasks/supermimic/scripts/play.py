"""Play a Go2 SuperMimic policy or inspect the velocity task."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.utils.torch import configure_torch_backends

from ..eval.playback import make_playback


@dataclass(frozen=True)
class PlayConfig:
    """Playback settings; use steps for a finite headless run."""

    terrain: Literal["flat", "rough"] | None = None
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: Path | None = None
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    steps: int | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    no_terminations: bool = False


def run_play(cfg: PlayConfig):
    if cfg.agent == "trained" and cfg.checkpoint_file is None:
        raise ValueError("Trained playback requires --checkpoint-file")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env, policy = make_playback(
        terrain=cfg.terrain,
        checkpoint_file=cfg.checkpoint_file,
        agent=cfg.agent,
        num_envs=cfg.num_envs,
        device=device,
        video=cfg.video,
        video_length=cfg.video_length,
        video_height=cfg.video_height,
        video_width=cfg.video_width,
        no_terminations=cfg.no_terminations,
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
    run_play(tyro.cli(PlayConfig))


if __name__ == "__main__":
    main()
