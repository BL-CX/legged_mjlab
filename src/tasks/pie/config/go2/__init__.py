"""Register the Unitree Go2 visual locomotion task."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.pie.rl import PIEOnPolicyRunner

from .env_cfgs import rough_env_cfg
from .rl_cfg import pie_runner_cfg


register_mjlab_task(
    task_id="go2-pie-rough",
    env_cfg=rough_env_cfg(),
    play_env_cfg=rough_env_cfg(play=True),
    rl_cfg=pie_runner_cfg(),
    runner_cls=PIEOnPolicyRunner,
)
