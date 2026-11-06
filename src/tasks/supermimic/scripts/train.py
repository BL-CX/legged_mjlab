"""Train SuperMimic.  Example:

.venv/bin/python src/tasks/supermimic/scripts/train.py go2-supermimic-flat --env.scene.num-envs 1024
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


# Under torchrun, pin each rank to its own GPU *before* CUDA/warp initialise: MuJoCo Warp and
# mjlab's simulation always run on the first visible device, so cfg.device alone is not enough.
# Skipped on a box with fewer GPUs than ranks (single-GPU smoke tests share cuda:0 via gloo).
def _n_gpus() -> int:
    try:
        return len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return 0


# A preset CUDA_VISIBLE_DEVICES list (e.g. "0,2,3" to skip a bad card) is indexed by LOCAL_RANK.
_preset = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
_avail = _preset if _preset else [str(i) for i in range(_n_gpus())]
if (
    "LOCAL_RANK" in os.environ
    and int(os.environ.get("LOCAL_WORLD_SIZE", "1")) <= len(_avail)
    and (not _preset or len(_preset) > 1)
):
    os.environ["CUDA_VISIBLE_DEVICES"] = _avail[int(os.environ["LOCAL_RANK"])]
    os.environ["SM_GPU_PINNED"] = "1"
    # Give each rank its own slice of cores (the training loop is single-thread CPU bound;
    # sharing cores between ranks adds jitter that lockstep all-reduces turn into waiting).
    try:
        cores = sorted(os.sched_getaffinity(0))
        n_local = int(os.environ["LOCAL_WORLD_SIZE"])
        per = len(cores) // n_local
        if per >= 2:
            r = int(os.environ["LOCAL_RANK"])
            os.sched_setaffinity(0, set(cores[r * per : (r + 1) * per]))
    except (OSError, ValueError):
        pass


def main() -> None:
    import mjlab
    import tyro
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    from src.tasks.supermimic.training.config import SuperMimicCfg, TrainConfig

    cfg = tyro.extras.overridable_config_cli(
        {
            task_id: (
                f"Train {task_id}.",
                TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id), run_name=task_id),
            )
            for task_id in list_tasks()
            if isinstance(load_rl_cfg(task_id), SuperMimicCfg)
        },
        config=mjlab.TYRO_FLAGS,
    )

    from src.tasks.supermimic.training.loop import Trainer

    Trainer(cfg).run()


if __name__ == "__main__":
    main()
