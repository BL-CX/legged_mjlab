"""Play and export a registered SuperMimic policy."""

# ruff: noqa: E402 -- support direct execution before importing repository packages.

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.supermimic.eval.export import export_checkpoint
from src.tasks.supermimic.eval.playback import make_playback
from src.tasks.supermimic.training.config import SuperMimicCfg


@dataclass(frozen=True)
class PlayConfig:
    """Playback settings; use steps for a finite headless run."""

    env: ManagerBasedRlEnvCfg
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: Path | None = None
    """Checkpoint to load; defaults to the latest local run."""
    run_name: str = "velocity"
    log_root: Path = Path("logs/supermimic")
    export: bool = True
    """Export TorchScript, ONNX and metadata before playback."""
    export_only: bool = False
    """Export without creating a playback environment."""
    output_dir: Path | None = None
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    steps: int | None = None
    video: bool = False
    video_length: int = 200


def _resolve_checkpoint(cfg: PlayConfig) -> Path:
    if cfg.checkpoint_file is not None:
        path = cfg.checkpoint_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return path

    log_dir = (cfg.log_root.expanduser() / cfg.run_name).resolve()
    runs = sorted((path for path in log_dir.glob("*") if path.is_dir()), reverse=True)
    # Keep pre-timestamp training directories usable after the migration.
    for run in [*runs, log_dir]:
        latest = run / "ckpt_latest.pt"
        if latest.is_file():
            return latest
        checkpoints = [
            path for path in run.glob("ckpt_*.pt") if path.is_file() and path.stem[5:].isdigit()
        ]
        if checkpoints:
            return max(checkpoints, key=lambda path: int(path.stem[5:]))
    raise FileNotFoundError(
        f"No checkpoints found in {log_dir}; check --log-root and --run-name or provide --checkpoint-file"
    )


def run_play(cfg: PlayConfig):
    if cfg.export_only and (cfg.agent != "trained" or not cfg.export):
        raise ValueError("--export-only requires --agent trained and --export True")
    checkpoint_file = None
    if cfg.agent == "trained":
        checkpoint_file = _resolve_checkpoint(cfg)
        print(f"[INFO] Loading checkpoint: {checkpoint_file}")
        if cfg.export:
            output_dir = cfg.output_dir or checkpoint_file.parent / "exported" / checkpoint_file.stem
            for name, path in export_checkpoint(checkpoint_file, output_dir).items():
                print(f"[INFO] Exported {name}: {path}")
        if cfg.export_only:
            return

    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env, policy = make_playback(
        env_cfg=cfg.env,
        checkpoint_file=checkpoint_file,
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
            task_id: (
                f"Play {task_id}.",
                PlayConfig(env=load_env_cfg(task_id, play=True), run_name=task_id),
            )
            for task_id in list_tasks()
            if isinstance(load_rl_cfg(task_id), SuperMimicCfg)
        },
        config=mjlab.TYRO_FLAGS,
    )
    run_play(cfg)


if __name__ == "__main__":
    main()
