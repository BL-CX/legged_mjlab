# Copyright (c) 2026-2027 zh
"""Checkpoint protocol and configuration snapshots."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .go2.env_cfg import EnvCfg

FRAMEWORK = "supermimic_velocity_v3"


def _update_config(target, values: dict) -> None:
    """Read declared fields from this protocol's JSON configuration."""
    if not isinstance(values, dict):
        raise ValueError("Configuration sections must be objects")
    declared = {item.name for item in fields(target)}
    for name, value in values.items():
        if name not in declared:
            raise ValueError(f"Unknown configuration field: {name}")
        current = getattr(target, name)
        if is_dataclass(current):
            _update_config(current, value)
        elif isinstance(current, Path):
            setattr(target, name, Path(value))
        else:
            if isinstance(current, tuple):
                value = tuple(value)
            setattr(target, name, deepcopy(value))


def robot_settings(env_cfg: EnvCfg) -> dict:
    from .go2.env_cfg import make_env_cfg

    native = make_env_cfg(env_cfg)
    settings = asdict(native.scene.entities["robot"])
    del settings["spec_fn"]
    settings["action_scale"] = native.actions["joint_pos"].scale
    return settings


def load_config(path: str | Path, *, snapshot: bool = False) -> tuple[EnvCfg, dict]:
    """Load environment settings from a current-protocol snapshot."""
    from .go2.env_cfg import EnvCfg

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Environment configuration must be an object")
    metadata = {name: value for name, value in payload.items() if name != "config"}
    if snapshot and metadata.get("framework") != FRAMEWORK:
        raise ValueError("Checkpoint uses an older training protocol; use its original checkout")
    cfg = EnvCfg.from_dict(payload.get("config", payload))
    if "robot" in metadata and metadata["robot"] != json.loads(json.dumps(robot_settings(cfg))):
        raise ValueError("Saved robot settings differ from the native Go2 configuration")
    return cfg, metadata
