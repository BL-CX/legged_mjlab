# Copyright (c) 2026-2027 zh
"""Load saved policies once for playback or explicit deployment export."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.string import resolve_matching_names_values

from ..config import FRAMEWORK, load_config
from ..config.go2.env_cfg import make_env_cfg
from ..config.go2.rl_cfg import TrainConfig
from ..envs.mdp.observations import OBS_TERMS
from ..policy.nets import Policy, PolicyCfg


def load_policy_checkpoint(path: str | Path):
    """Validate a CPU checkpoint and its adjacent task snapshot."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FRAMEWORK:
        raise ValueError("Checkpoint uses an older training protocol; use its original checkout")
    cfg, snapshot = load_config(path.parent / "task_config.json", snapshot=True)
    trained = TrainConfig.from_dict(checkpoint["config"])
    if checkpoint.get("spec_hash") != snapshot.get("spec_hash"):
        raise ValueError("Checkpoint and task snapshot metadata mismatch")
    if json.dumps(asdict(trained.env), sort_keys=True) != json.dumps(asdict(cfg), sort_keys=True):
        raise ValueError("Checkpoint and task snapshot configuration mismatch")
    return cfg, snapshot, checkpoint


def checkpoint_actor(checkpoint: dict):
    """Construct the saved policy at the caller's intended RNG boundary."""
    actor = Policy(
        PolicyCfg(**checkpoint["config"]["policy"]), history_length=checkpoint["config"]["env"]["history_length"]
    )
    actor.load_state_dict(checkpoint["policy"], strict=True)
    return actor.eval().requires_grad_(False)


def export_checkpoint(checkpoint_path: str | Path, directory: str | Path):
    """Write policy.pt, policy.onnx, and policy.json into the requested directory."""
    import onnx

    cfg, _, checkpoint = load_policy_checkpoint(checkpoint_path)
    actor = checkpoint_actor(checkpoint)
    native_cfg = make_env_cfg(cfg)
    robot_cfg = native_cfg.scene.entities["robot"]
    joint_names = Entity(robot_cfg).joint_names
    _, _, default_joint_positions = resolve_matching_names_values(robot_cfg.init_state.joint_pos, joint_names)
    action_scales = [native_cfg.actions["joint_pos"].scale] * len(joint_names)
    iteration = checkpoint["iteration"]
    layout = [{"name": name, "width": width} for name, width in OBS_TERMS]
    information = {
        "format": "supermimic_velocity_policy_v2",
        "spec_hash": checkpoint["spec_hash"],
        "iteration": iteration,
        "input_name": "obs_history",
        "output_name": "actions",
        "dtype": "float32",
        "input_shape": [None, actor.history_length * actor.observation_dim],
        "output_shape": [None, actor.action_dim],
        "observation_dim": actor.observation_dim,
        "observation_layout": layout,
        "history_length": actor.history_length,
        "history_order": actor.history_order,
        "history_reset": "repeat_initial_frame",
        "normalization": {
            "in_graph": True,
            "method": "fixed_physical_scales",
            "term_scales": checkpoint["config"]["policy"]["obs_term_scale"],
        },
        "action_joint_names": list(joint_names),
        "default_joint_positions": default_joint_positions,
        "action_scales": action_scales,
        "action_semantics": "unscaled_joint_position_offsets",
        "control_dt_seconds": cfg.timestep * cfg.decimation,
        "command_frame": "body_xy_world_yaw",
        "gait_phase": {
            "enabled": True,
            "period_seconds": cfg.reward.gait.period,
            "standing_command_threshold": cfg.reward.gait.command_threshold,
        },
    }
    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "pt": destination / "policy.pt",
        "onnx": destination / "policy.onnx",
        "metadata": destination / "policy.json",
    }
    with TemporaryDirectory(prefix=".policy_export_", dir=destination) as temporary:
        pending = Path(temporary)
        torch.jit.script(actor).save(str(pending / "policy.pt"))
        torch.onnx.export(
            actor,
            (torch.zeros(1, actor.history_length * actor.observation_dim),),
            str(pending / "policy.onnx"),
            export_params=True,
            opset_version=18,
            dynamo=False,
            input_names=["obs_history"],
            output_names=["actions"],
            external_data=False,
            dynamic_axes={"obs_history": {0: "batch"}, "actions": {0: "batch"}},
        )
        model = onnx.load(pending / "policy.onnx")
        for name, value in information.items():
            field = model.metadata_props.add()
            field.key, field.value = name, json.dumps(value, ensure_ascii=False, allow_nan=False)
        onnx.checker.check_model(model)
        onnx.save(model, pending / "policy.onnx")
        (pending / "policy.json").write_text(
            json.dumps(information, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )
        for path in paths.values():
            os.replace(pending / path.name, path)
    return paths
