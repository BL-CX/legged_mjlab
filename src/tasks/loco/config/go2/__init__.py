"""Register Unitree Go2 HIM and HIM-AMP tasks."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.loco.rsl_rl.runners import HIMAMPOnPolicyRunner, HIMOnPolicyRunner

from .env_cfgs import flat_env_cfg, rough_env_cfg
from .rl_cfg import him_amp_runner_cfg, him_runner_cfg


register_mjlab_task(
    task_id="go2-him-rough",
    env_cfg=rough_env_cfg(),
    play_env_cfg=rough_env_cfg(play=True),
    rl_cfg=him_runner_cfg(terrain="rough"),
    runner_cls=HIMOnPolicyRunner,
)

register_mjlab_task(
    task_id="go2-him-flat",
    env_cfg=flat_env_cfg(),
    play_env_cfg=flat_env_cfg(play=True),
    rl_cfg=him_runner_cfg(terrain="flat"),
    runner_cls=HIMOnPolicyRunner,
)

register_mjlab_task(
    task_id="go2-him-amp-rough",
    env_cfg=rough_env_cfg(),
    play_env_cfg=rough_env_cfg(play=True),
    rl_cfg=him_amp_runner_cfg(terrain="rough"),
    runner_cls=HIMAMPOnPolicyRunner,
)

register_mjlab_task(
    task_id="go2-him-amp-flat",
    env_cfg=flat_env_cfg(),
    play_env_cfg=flat_env_cfg(play=True),
    rl_cfg=him_amp_runner_cfg(terrain="flat"),
    runner_cls=HIMAMPOnPolicyRunner,
)
