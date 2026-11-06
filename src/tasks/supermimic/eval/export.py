"""Load saved policies once for playback or explicit deployment export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.string import resolve_matching_names_values

from ..envs.mdp.observations import OBS_TERMS
from ..policy.nets import Policy
from ..training.config import CHECKPOINT_FORMAT


def load_policy_checkpoint(path: str | Path):
    """Load weights and their native environment configuration from one file."""
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint format; use an aligned quadruped checkpoint")
    return checkpoint


def checkpoint_actor(checkpoint: dict):
    """Construct the saved policy at the caller's intended RNG boundary."""
    cfg = checkpoint["config"]
    actor = Policy(cfg.agent.policy, history_length=cfg.env.observations["history"].history_length)
    actor.load_state_dict(checkpoint["policy"], strict=True)
    return actor.eval().requires_grad_(False)


def export_checkpoint(checkpoint_path: str | Path, directory: str | Path):
    """Write policy.pt, policy.onnx, and policy.json into the requested directory."""
    import onnx

    checkpoint = load_policy_checkpoint(checkpoint_path)
    cfg = checkpoint["config"].env
    actor = checkpoint_actor(checkpoint)
    robot_cfg = cfg.scene.entities["robot"]
    joint_names = Entity(robot_cfg).joint_names
    _, _, default_joint_positions = resolve_matching_names_values(robot_cfg.init_state.joint_pos, joint_names)
    scale = cfg.actions["joint_pos"].scale
    if isinstance(scale, dict):
        # Match native action defaults and resolve scales in the robot joint order.
        action_scales = [1.0] * len(joint_names)
        indices, _, values = resolve_matching_names_values(scale, joint_names)
        for index, value in zip(indices, values, strict=True):
            action_scales[index] = value
    else:
        action_scales = [scale] * len(joint_names)
    iteration = checkpoint["iteration"]
    layout = [{"name": name, "width": width} for name, width in OBS_TERMS]
    information = {
        "format": "supermimic_velocity_policy_v3",
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
            "method": "running_mean_variance",
            "epsilon": 1e-6,
            "clip": 10.0,
            "history": {name: getattr(actor.normalizer, name).tolist() for name in ("mean", "var", "count")},
            "current_frame": {
                name: getattr(actor.frame_normalizer, name).tolist() for name in ("mean", "var", "count")
            },
        },
        "action_joint_names": list(joint_names),
        "default_joint_positions": default_joint_positions,
        "action_scales": action_scales,
        "action_semantics": "unscaled_joint_position_offsets",
        "control_dt_seconds": cfg.sim.mujoco.timestep * cfg.decimation,
        "command_frame": "body_xy_world_yaw",
        "gait_phase": {
            "enabled": True,
            "period_seconds": cfg.commands["velocity"].reward.gait.period,
            "standing_command_threshold": cfg.commands["velocity"].reward.gait.command_threshold,
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
            field.key, field.value = (
                name,
                json.dumps(value, ensure_ascii=False, allow_nan=False),
            )
        onnx.checker.check_model(model)
        onnx.save(model, pending / "policy.onnx")
        (pending / "policy.json").write_text(
            json.dumps(information, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in paths.values():
            os.replace(pending / path.name, path)
    return paths
