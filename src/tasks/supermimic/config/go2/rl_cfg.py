"""Unitree Go2 SuperMimic learning defaults."""

from ...training.config import SuperMimicCfg


def supermimic_cfg() -> SuperMimicCfg:
    """Create fresh world-model and policy-learning settings for Go2."""
    return SuperMimicCfg()
