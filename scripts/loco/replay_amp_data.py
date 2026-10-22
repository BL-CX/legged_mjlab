"""Replay AMP motion files with MuJoCo forward kinematics (no physics stepping).

    python scripts/loco/replay_amp_data.py --viewer native
    python scripts/loco/replay_amp_data.py --viewer viser --clip 3 --loop
    python scripts/loco/replay_amp_data.py --motion path/to/forward.txt --speed 0.5
"""

# ruff: noqa: E402 -- prefer this repository over other editable src packages.

import os
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import mujoco
import mujoco.viewer
import numpy as np
import torch
import tyro

from src import SRC_PATH
from src.tasks.loco.rsl_rl.datasets.amp_motion_loader import AMPMotionLoader


@dataclass
class ReplayConfig:
    motion: Path = SRC_PATH / "assets/motions/mocap_motions_go2"
    """Motion JSON/text file or directory."""
    mjcf: Path = SRC_PATH / "assets/robots/unitree_go2/xmls/scene_go2.xml"
    viewer: Literal["auto", "native", "viser", "none"] = "auto"
    clip: int | None = None
    """Zero-based index; None plays every motion."""
    dt: float = 0.02
    speed: float = 1.0
    loop: bool = False
    reorder: bool = False
    """Use the loader's conversion from leg-grouped to joint-type-grouped order."""
    list_motions: bool = False
    max_steps: int | None = None
    real_time: bool = True
    port: int = 8080
    """Viser server port."""


def write_frame(model, data, loader, frame, joint_ids):
    """MuJoCo free-joint qvel uses WORLD linear and BODY angular velocity."""
    pose = frame[0].numpy()
    if not np.isfinite(pose).all():
        raise ValueError("Non-finite interpolated motion frame; refusing to render")
    free_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_ids) != 1:
        raise ValueError("Replay expects one free-root robot")
    qadr, vadr = model.jnt_qposadr[free_ids[0]], model.jnt_dofadr[free_ids[0]]
    xyzw = loader.get_root_rot(pose)
    wxyz = np.roll(xyzw, 1).copy()
    norm = np.linalg.norm(wxyz)
    if norm < 1e-8:
        raise ValueError("Invalid zero root quaternion")
    wxyz /= norm
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, wxyz)
    data.qpos[qadr:qadr + 3] = loader.get_root_pos(pose)
    data.qpos[qadr + 3:qadr + 7] = wxyz
    data.qpos[model.jnt_qposadr[joint_ids]] = loader.get_joint_pose(pose)
    data.qvel[vadr:vadr + 3] = rotation.reshape(3, 3) @ loader.get_linear_vel_batch(pose[None])[0]
    data.qvel[vadr + 3:vadr + 6] = loader.get_angular_vel_batch(pose[None])[0]
    data.qvel[model.jnt_dofadr[joint_ids]] = loader.get_joint_vel_batch(pose[None])[0]
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)


def run_replay(cfg: ReplayConfig):
    if not np.isfinite([cfg.dt, cfg.speed]).all() or min(cfg.dt, cfg.speed) <= 0:
        raise ValueError("dt and speed must be finite and positive")
    if cfg.max_steps is not None and cfg.max_steps < 1:
        raise ValueError("max-steps must be positive")
    path = cfg.motion.expanduser().resolve()
    files = sorted([*path.glob("*.txt"), *path.glob("*.json")]) if path.is_dir() else [path]
    if not files or any(not p.is_file() for p in files):
        raise FileNotFoundError(f"No motion files found at {path}")
    loader = AMPMotionLoader("cpu", cfg.dt, motion_files=[str(p) for p in files], reorder=cfg.reorder)
    for i, (file, duration) in enumerate(zip(files, loader.trajectory_lens)):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Clip must have positive duration: {file}")
        print(f"[{i}] {file.name}: {duration:.3f}s")
    if cfg.clip is not None and not 0 <= cfg.clip < loader.num_motions:
        raise ValueError(f"clip must be in [0, {loader.num_motions - 1}]")
    if cfg.list_motions:
        return

    model = mujoco.MjModel.from_xml_path(str(cfg.mjcf.expanduser().resolve()))
    data = mujoco.MjData(model)
    legs, joints = ("FL", "FR", "RL", "RR"), ("hip", "thigh", "calf")
    names = ([f"{leg}_{joint}_joint" for joint in joints for leg in legs] if cfg.reorder
             else [f"{leg}_{joint}_joint" for leg in legs for joint in joints])
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
    if (joint_ids < 0).any():
        raise ValueError("MJCF is missing the expected Go2 joints")
    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if root_body < 0:
        raise ValueError("MJCF is missing the Go2 base_link body")
    kind = cfg.viewer
    if kind == "auto":
        kind = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
    indices = range(loader.num_motions) if cfg.clip is None else [cfg.clip]
    steps = 0
    with ExitStack() as stack:
        if kind == "native":
            viewer = stack.enter_context(mujoco.viewer.launch_passive(model, data))
            viewer.cam.distance, viewer.cam.elevation = 3.0, -25.0
            def show():
                with viewer.lock():
                    viewer.cam.lookat[:] = data.xpos[root_body]
                viewer.sync()
            running = viewer.is_running
        elif kind == "viser":
            import viser
            from mjlab.viewer.viser.scene import ViserMujocoScene
            server = viser.ViserServer(port=cfg.port)
            stack.callback(server.stop)
            scene = ViserMujocoScene.create(server, model, num_envs=1)
            scene.create_visualization_gui(show_debug_viz_control=False)
            show = lambda: scene.update_from_mjdata(data)
            running = lambda: True
        else:
            show = lambda: None
            running = lambda: True
        while running():
            for index in indices:
                print(f"[PLAY] {index}: {files[index].name}")
                mujoco.mj_resetData(model, data)
                duration = loader.trajectory_lens[index]
                for t in np.append(np.arange(0.0, duration, cfg.dt), duration):
                    if not running():
                        return
                    start = time.perf_counter()
                    frame = loader.get_full_frame_at_time_batch(np.array([index]), np.array([t]))
                    write_frame(model, data, loader, frame, joint_ids)
                    data.time = float(t)
                    show()
                    steps += 1
                    if cfg.max_steps is not None and steps >= cfg.max_steps:
                        print(f"[REPLAY] displayed {steps} dataset frames")
                        return
                    if cfg.real_time:
                        time.sleep(max(0.0, cfg.dt / cfg.speed - (time.perf_counter() - start)))
            if not cfg.loop:
                break
    print(f"[REPLAY] displayed {steps} dataset frames")


if __name__ == "__main__":
    run_replay(tyro.cli(ReplayConfig))
