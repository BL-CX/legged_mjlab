# Copyright (c) 2026-2027 zh
"""Native recorder hooks for replay transitions and terminal states."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

from .replay_buffer import ReplayBuffer


class ReplayRecorder(RecorderTerm):
    """Remember reset states and write each transition before its episode is reset."""

    def __init__(self, cfg: ReplayRecorderCfg, env):
        super().__init__(cfg, env)
        spec = self.command.model.spec
        self.buffer = ReplayBuffer(
            env.num_envs, cfg.params["capacity"], spec.state_dim, spec.act_dim, spec.xi_dim, env.device
        )
        self.prev_s = torch.zeros(env.num_envs, spec.state_dim, device=env.device)
        self.prev_xi = torch.zeros(env.num_envs, spec.xi_dim, device=env.device)
        self.ep_start = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.clean_action = None

    @property
    def command(self):
        return self._env.command_manager.get_term("velocity")

    def _remember(self, ids):
        state = self.command.state
        self.prev_s[ids] = self.command.model.spec.pack(state)[ids]
        self.prev_xi[ids] = state.context[ids]

    def record_post_reset(self, env_ids):
        # Explicit viewer resets also start a new replay episode.
        self.buffer.end_episode(env_ids[~self.ep_start[env_ids]])
        self._remember(env_ids)
        self.ep_start[env_ids] = True

    def _write_rows(self, ids, done):
        env = self._env
        actions = self.clean_action if self.clean_action is not None else env.action_manager.action
        self.buffer.add(
            ids,
            s=self.prev_s[ids],
            a=actions[ids],
            a_exec=env.action_manager.action[ids],
            xi=self.prev_xi[ids],
            r=env.reward_buf[ids],
            done=done,
            terminal=torch.zeros_like(done, dtype=torch.bool),
            ep_start=self.ep_start[ids],
        )
        self.ep_start[ids] = False

    def record_pre_reset(self, env_ids):
        env = self._env
        labels = env.reset_terminated[env_ids].float()
        self._write_rows(env_ids, labels)
        zeros_a = torch.zeros(env_ids.numel(), self.command.model.spec.act_dim, device=env.device)
        self.buffer.add(
            env_ids,
            s=self.command.model.spec.pack(self.command.final_state)[env_ids],
            a=zeros_a,
            a_exec=zeros_a,
            xi=self.prev_xi[env_ids],
            r=torch.zeros_like(labels),
            done=labels,
            terminal=torch.ones_like(labels, dtype=torch.bool),
            ep_start=torch.zeros_like(labels, dtype=torch.bool),
        )

    def record_post_step(self):
        ids = (~self._env.reset_buf).nonzero().squeeze(-1)
        if ids.numel():
            self._write_rows(ids, torch.zeros(ids.numel(), device=self._env.device))
            self._remember(ids)


@dataclass
class ReplayRecorderCfg(RecorderTermCfg):
    func: type = ReplayRecorder
    params: dict = field(default_factory=lambda: {"capacity": 1000})
