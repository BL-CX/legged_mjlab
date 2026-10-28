# Copyright (c) 2026-2027 zh
"""Expert motion loader preserving the existing AMP sampling behavior."""

import json

import numpy as np
import torch

from .motion_utils import QuaternionNormalize, quaternion_slerp, standardize_quaternion


class AMPMotionLoader:
    """Sample expert transitions from weighted motion clips."""

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    TAR_TOE_VEL_LOCAL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    TAR_TOE_VEL_LOCAL_START_IDX = JOINT_VEL_END_IDX
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=None,
        reorder=False,
    ):
        """Load clips and optionally cache transitions at the requested interval."""
        if not motion_files:
            raise ValueError("AMP motion_files is empty; configure at least one motion JSON file")
        self.device = device
        self.time_between_frames = time_between_frames

        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.asarray(motion_json["Frames"], dtype=np.float32)

                if reorder:
                    motion_data = self.reorder_from_pybullet_to_isaac(motion_data)

                for f_i in range(motion_data.shape[0]):
                    root_rot = AMPMotionLoader.get_root_rot(motion_data[f_i])
                    root_rot = QuaternionNormalize(root_rot)
                    root_rot = standardize_quaternion(root_rot)
                    motion_data[
                        f_i,
                        AMPMotionLoader.POS_SIZE : (
                            AMPMotionLoader.POS_SIZE + AMPMotionLoader.ROT_SIZE
                        ),
                    ] = root_rot

                self.trajectories.append(
                    torch.tensor(
                        motion_data[
                            :, AMPMotionLoader.ROOT_ROT_END_IDX : AMPMotionLoader.JOINT_VEL_END_IDX
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : AMPMotionLoader.JOINT_VEL_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(
            self.trajectory_weights
        )
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames
            )
            print(f"Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def reorder_from_pybullet_to_isaac(self, motion_data):
        """Convert leg-grouped joint data to joint-type order."""
        root_pos = AMPMotionLoader.get_root_pos_batch(motion_data)
        root_rot = AMPMotionLoader.get_root_rot_batch(motion_data)

        joint_pos = AMPMotionLoader.get_joint_pose_batch(motion_data)
        joint_pos = joint_pos.reshape(-1, 4, 3).transpose(0, 2, 1).reshape(-1, 12)

        foot_pos = AMPMotionLoader.get_tar_toe_pos_local_batch(motion_data)

        lin_vel = AMPMotionLoader.get_linear_vel_batch(motion_data)
        ang_vel = AMPMotionLoader.get_angular_vel_batch(motion_data)

        joint_vel = AMPMotionLoader.get_joint_vel_batch(motion_data)
        joint_vel = joint_vel.reshape(-1, 4, 3).transpose(0, 2, 1).reshape(-1, 12)

        foot_vel = AMPMotionLoader.get_tar_toe_vel_local_batch(motion_data)

        return np.hstack(
            [root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel, foot_vel]
        )

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True
        )

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = (
            self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        )
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0] - 1
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs] - 1
        idx_low, idx_high = np.floor(p * n).astype(int), np.ceil(p * n).astype(int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0] - 1
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs] - 1
        idx_low, idx_high = np.floor(p * n).astype(int), np.ceil(p * n).astype(int)
        all_frame_pos_starts = torch.zeros(
            len(traj_idxs), AMPMotionLoader.POS_SIZE, device=self.device
        )
        all_frame_pos_ends = torch.zeros(
            len(traj_idxs), AMPMotionLoader.POS_SIZE, device=self.device
        )
        all_frame_rot_starts = torch.zeros(
            len(traj_idxs), AMPMotionLoader.ROT_SIZE, device=self.device
        )
        all_frame_rot_ends = torch.zeros(
            len(traj_idxs), AMPMotionLoader.ROT_SIZE, device=self.device
        )
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs),
            AMPMotionLoader.JOINT_VEL_END_IDX - AMPMotionLoader.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs),
            AMPMotionLoader.JOINT_VEL_END_IDX - AMPMotionLoader.JOINT_POSE_START_IDX,
            device=self.device,
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = AMPMotionLoader.get_root_pos_batch(
                trajectory[idx_low[traj_mask]]
            )
            all_frame_pos_ends[traj_mask] = AMPMotionLoader.get_root_pos_batch(
                trajectory[idx_high[traj_mask]]
            )
            all_frame_rot_starts[traj_mask] = AMPMotionLoader.get_root_rot_batch(
                trajectory[idx_low[traj_mask]]
            )
            all_frame_rot_ends[traj_mask] = AMPMotionLoader.get_root_rot_batch(
                trajectory[idx_high[traj_mask]]
            )
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPMotionLoader.JOINT_POSE_START_IDX : AMPMotionLoader.JOINT_VEL_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPMotionLoader.JOINT_POSE_START_IDX : AMPMotionLoader.JOINT_VEL_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Interpolate full frames with quaternion SLERP."""

        root_pos0, root_pos1 = AMPMotionLoader.get_root_pos(frame0), AMPMotionLoader.get_root_pos(
            frame1
        )
        root_rot0, root_rot1 = AMPMotionLoader.get_root_rot(frame0), AMPMotionLoader.get_root_rot(
            frame1
        )
        joints0, joints1 = AMPMotionLoader.get_joint_pose(frame0), AMPMotionLoader.get_joint_pose(
            frame1
        )
        tar_toe_pos_0, tar_toe_pos_1 = AMPMotionLoader.get_tar_toe_pos_local(
            frame0
        ), AMPMotionLoader.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = AMPMotionLoader.get_linear_vel(
            frame0
        ), AMPMotionLoader.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = AMPMotionLoader.get_angular_vel(
            frame0
        ), AMPMotionLoader.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = AMPMotionLoader.get_joint_vel(
            frame0
        ), AMPMotionLoader.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = quaternion_slerp(
            root_rot0.unsqueeze(0),
            root_rot1.unsqueeze(0),
            torch.tensor([blend], device=self.device).unsqueeze(-1),
        ).squeeze(0)
        blend_root_rot = standardize_quaternion(blend_root_rot)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat(
            [
                blend_root_pos,
                blend_root_rot,
                blend_joints,
                blend_tar_toe_pos,
                blend_linear_vel,
                blend_angular_vel,
                blend_joints_vel,
            ]
        )

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = torch.cat(
                    [
                        self.preloaded_s[
                            idxs,
                            AMPMotionLoader.JOINT_POSE_START_IDX : AMPMotionLoader.JOINT_POSE_END_IDX,
                        ],
                        self.preloaded_s[
                            idxs,
                            AMPMotionLoader.LINEAR_VEL_START_IDX : AMPMotionLoader.JOINT_VEL_END_IDX,
                        ],
                    ],
                    dim=-1,
                )
                s_next = torch.cat(
                    [
                        self.preloaded_s_next[
                            idxs,
                            AMPMotionLoader.JOINT_POSE_START_IDX : AMPMotionLoader.JOINT_POSE_END_IDX,
                        ],
                        self.preloaded_s_next[
                            idxs,
                            AMPMotionLoader.LINEAR_VEL_START_IDX : AMPMotionLoader.JOINT_VEL_END_IDX,
                        ],
                    ],
                    dim=-1,
                )
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(
                        self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames)
                    )

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return 30

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    @staticmethod
    def get_root_pos(pose):
        return pose[AMPMotionLoader.ROOT_POS_START_IDX : AMPMotionLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_pos_batch(poses):
        return poses[:, AMPMotionLoader.ROOT_POS_START_IDX : AMPMotionLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_rot(pose):
        return pose[AMPMotionLoader.ROOT_ROT_START_IDX : AMPMotionLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_root_rot_batch(poses):
        return poses[:, AMPMotionLoader.ROOT_ROT_START_IDX : AMPMotionLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_joint_pose(pose):
        return pose[AMPMotionLoader.JOINT_POSE_START_IDX : AMPMotionLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses):
        return poses[:, AMPMotionLoader.JOINT_POSE_START_IDX : AMPMotionLoader.JOINT_POSE_END_IDX]

    def get_tar_toe_pos_local(pose):
        return pose[
            AMPMotionLoader.TAR_TOE_POS_LOCAL_START_IDX : AMPMotionLoader.TAR_TOE_POS_LOCAL_END_IDX
        ]

    @staticmethod
    def get_tar_toe_pos_local_batch(poses):
        return poses[
            :,
            AMPMotionLoader.TAR_TOE_POS_LOCAL_START_IDX : AMPMotionLoader.TAR_TOE_POS_LOCAL_END_IDX,
        ]

    def get_linear_vel(pose):
        return pose[AMPMotionLoader.LINEAR_VEL_START_IDX : AMPMotionLoader.LINEAR_VEL_END_IDX]

    @staticmethod
    def get_linear_vel_batch(poses):
        return poses[:, AMPMotionLoader.LINEAR_VEL_START_IDX : AMPMotionLoader.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[AMPMotionLoader.ANGULAR_VEL_START_IDX : AMPMotionLoader.ANGULAR_VEL_END_IDX]

    @staticmethod
    def get_angular_vel_batch(poses):
        return poses[:, AMPMotionLoader.ANGULAR_VEL_START_IDX : AMPMotionLoader.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        return pose[AMPMotionLoader.JOINT_VEL_START_IDX : AMPMotionLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses):
        return poses[:, AMPMotionLoader.JOINT_VEL_START_IDX : AMPMotionLoader.JOINT_VEL_END_IDX]

    def get_tar_toe_vel_local(pose):
        return pose[
            AMPMotionLoader.TAR_TOE_VEL_LOCAL_START_IDX : AMPMotionLoader.TAR_TOE_VEL_LOCAL_END_IDX
        ]

    @staticmethod
    def get_tar_toe_vel_local_batch(poses):
        return poses[
            :,
            AMPMotionLoader.TAR_TOE_VEL_LOCAL_START_IDX : AMPMotionLoader.TAR_TOE_VEL_LOCAL_END_IDX,
        ]
