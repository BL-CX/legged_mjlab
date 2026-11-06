"""Train SuperMimic.  Example:

.venv/bin/python src/tasks/supermimic/scripts/train.py go2-supermimic-flat --env.scene.num-envs 1024
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


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
