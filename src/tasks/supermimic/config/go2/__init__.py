# Copyright (c) 2026-2027 zh
"""按 MJLab 的环境、播放环境、算法配置及 runner 四项注册任务。"""

from mjlab.tasks.registry import register_mjlab_task

from ...rl.runner import Go2SuperMimicRunner
from .env_cfgs import flat_env_cfg, rough_env_cfg
from .rl_cfg import supermimic_runner_cfg


for terrain, factory in (("flat", flat_env_cfg), ("rough", rough_env_cfg)):
    register_mjlab_task(
        task_id=f"go2-supermimic-{terrain}",
        env_cfg=factory(),
        play_env_cfg=factory(play=True),
        rl_cfg=supermimic_runner_cfg(terrain),
        runner_cls=Go2SuperMimicRunner,
    )
