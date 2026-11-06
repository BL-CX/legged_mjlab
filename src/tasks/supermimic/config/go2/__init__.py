"""Register Unitree Go2 SuperMimic velocity tasks."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import flat_env_cfg, rough_env_cfg
from .rl_cfg import supermimic_cfg

register_mjlab_task(
    task_id="go2-supermimic-flat",
    env_cfg=flat_env_cfg(),
    play_env_cfg=flat_env_cfg(play=True),
    rl_cfg=supermimic_cfg(),
)

register_mjlab_task(
    task_id="go2-supermimic-rough",
    env_cfg=rough_env_cfg(),
    play_env_cfg=rough_env_cfg(play=True),
    rl_cfg=supermimic_cfg(),
)
