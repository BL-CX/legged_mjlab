# Copyright (c) 2026-2027 zh
"""SuperMimic 的训练、播放和部署命令；入口不进入 RSL PPO 包装器。"""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Literal

import mjlab
from mjlab.tasks.registry import list_tasks
from mjlab.utils.torch import configure_torch_backends
import torch
import tyro

from .config.train_cfg import SuperMimicTrainConfig, TASK_MODES
from .rl.training import launch_training


def _choose_task():
    """只允许本算法任务；任务已由 src.tasks 的导入器正式注册。"""
    choices = sorted(set(list_tasks()) & set(TASK_MODES))
    return tyro.cli(
        tyro.extras.literal_type_from_choices(choices), add_help=False,
        return_unknown_args=True, config=mjlab.TYRO_FLAGS,
    )


def train_main():
    task, remaining = _choose_task()
    cfg = tyro.cli(SuperMimicTrainConfig, args=remaining,
                   default=SuperMimicTrainConfig.from_task(task),
                   prog=f"{sys.argv[0]} {task}", config=mjlab.TYRO_FLAGS)
    launch_training(task, cfg)


@dataclass(frozen=True)
class PlayConfig:
    """完整策略只需检查点与同目录任务快照；steps 用于无窗口验证。"""

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


def run_play(task_id: str, cfg: PlayConfig):
    from .rl.playback import make_playback_env

    if task_id not in TASK_MODES:
        raise ValueError(f"Unknown SuperMimic task: {task_id}")
    for name in ("num_envs", "steps", "video_length", "video_height", "video_width"):
        value = getattr(cfg, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if cfg.agent == "trained" and cfg.checkpoint_file is None:
        raise ValueError("Trained playback requires --checkpoint-file")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env, policy = make_playback_env(
        task_id, checkpoint_file=cfg.checkpoint_file, agent=cfg.agent,
        num_envs=cfg.num_envs, device=device, video=cfg.video, video_length=cfg.video_length,
        video_height=cfg.video_height, video_width=cfg.video_width,
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


def play_main():
    task, remaining = _choose_task()
    cfg = tyro.cli(PlayConfig, args=remaining, prog=f"{sys.argv[0]} {task}", config=mjlab.TYRO_FLAGS)
    run_play(task, cfg)


def export_main():
    from .rl.deployment import export_checkpoint

    def export(checkpoint_file: Path, output_dir: Path):
        """导出完整历史策略的 TorchScript、ONNX 和部署元数据。"""
        paths = export_checkpoint(checkpoint_file, output_dir)
        for name, path in paths.items():
            print(f"[INFO] Exported {name}: {path}")

    tyro.cli(export, config=mjlab.TYRO_FLAGS)
