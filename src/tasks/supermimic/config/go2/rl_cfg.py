# Copyright (c) 2026-2027 zh
"""用 MJLab 配置工厂注册 SuperMimic 的算法和运行预算。"""

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg

from ...rl.core.config import ExecutionConfig, SuperMimicConfig


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
    """基础运行字段沿用 MJLab，采集频率由 algorithm.collect_steps 定义。"""

    class_name: str = "src.tasks.supermimic.rl.runner:Go2SuperMimicRunner"
    algorithm: SuperMimicConfig = field(default_factory=SuperMimicConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


def supermimic_runner_cfg(terrain: str = "flat") -> SuperMimicRunnerCfg:
    """返回独立配置；训练使用专用入口和原版 replay/BPTT 协议。"""
    return SuperMimicRunnerCfg(
        seed=0,
        max_iterations=40000,
        save_interval=1000,
        experiment_name=f"go2_supermimic_{terrain}",
        logger="tensorboard",
        clip_actions=None,
        upload_model=False,
    )
