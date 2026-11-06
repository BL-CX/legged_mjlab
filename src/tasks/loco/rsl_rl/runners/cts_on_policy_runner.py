# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
from datetime import timedelta
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from ..algorithms import CTS
from ..modules import CTSActorCritic
from rsl_rl.env import VecEnv


class CTSOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Derive dimensions from named observations, not ambiguous env attributes.
        observations = self.env.get_observations()
        history = observations["actor"]
        critic = observations["critic"]
        if history.ndim != 3 or history.shape[1] < 2 or critic.ndim != 2:
            raise ValueError("CTS requires actor [N,H,F] with H >= 2 and critic [N,C]")
        self.num_one_step_obs = history.shape[-1]
        self.num_actor_obs = history.shape[1] * self.num_one_step_obs
        self.num_critic_obs = critic.shape[-1]
        self.num_privileged_obs = observations["privileged"].shape[-1]

        # Initialize policy network
        model=CTSActorCritic(
            self.num_actor_obs,
            self.num_privileged_obs,  # Domain parameters, foot contacts and terrain.
            self.num_critic_obs,
            self.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg).to(self.device)

        # Initialize algorithm
        self.alg=CTS(model, self.env.num_envs, device=self.device, **self.alg_cfg)

        # Training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.num_actor_obs],
            [self.num_privileged_obs],
            [self.num_critic_obs],
            [self.env.num_actions])

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        # Index of the next policy update.
        self.current_learning_iteration = 0

    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        observations = self.env.get_observations()
        obs = observations["actor"].flatten(1).to(self.device)
        critic_obs_buf = observations["critic"].to(self.device)
        privileged_buf = observations["privileged"].to(self.device)
        self.alg.model.train() # switch to train mode (for dropout for example)

        ep_infos = []
        teacher_rewbuffer = deque(maxlen=100)
        teacher_lenbuffer = deque(maxlen=100)
        student_rewbuffer = deque(maxlen=100)
        student_lenbuffer = deque(maxlen=100)

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        log_start_time = self.tot_time
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_buf,critic_obs_buf)
                    observations, rewards, dones, infos = self.env.step(actions)
                    obs = observations["actor"].flatten(1).to(self.device)
                    critic_obs_buf = observations["critic"].to(self.device)
                    privileged_buf = observations["privileged"].to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        elif 'log' in infos:
                            ep_infos.append(infos['log'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if new_ids.shape[0]:
                            ti = self.alg.teacher_env_idxs
                            teacher_ids = new_ids[torch.isin(new_ids, ti)]
                            student_ids = new_ids[~torch.isin(new_ids, ti)]
                            teacher_rewbuffer.extend(cur_reward_sum[teacher_ids].cpu().numpy().tolist())
                            teacher_lenbuffer.extend(cur_episode_length[teacher_ids].cpu().numpy().tolist())
                            student_rewbuffer.extend(cur_reward_sum[student_ids].cpu().numpy().tolist())
                            student_lenbuffer.extend(cur_episode_length[student_ids].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop


                self.alg.compute_returns(obs, privileged_buf,critic_obs_buf)
            
            mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it + 1
            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        if self.log_dir is not None:
            if self.current_learning_iteration > start_iter:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration - 1)))
            self.writer.flush()

    def log(self, locs, width=80, pad=40):
        """Log training information to console and TensorBoard."""
        loss_dict = {
            'value_function': locs['mean_value_loss'],
            'surrogate': locs['mean_surrogate_loss'],
            'entropy': locs['mean_entropy_loss'],
            'latent': locs['mean_latent_loss'],
        }
        reward_buffers = {
            'teacher': (locs['teacher_rewbuffer'], locs['teacher_lenbuffer']),
            'student': (locs['student_rewbuffer'], locs['student_lenbuffer']),
        }
        it = locs['it']
        collection_time = locs['collection_time']
        learn_time = locs['learn_time']
        iteration_time = collection_time + learn_time
        collection_size = self.num_steps_per_env * self.env.num_envs
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time
        fps = int(collection_size / max(iteration_time, 1e-8))
        mean_std = self.alg.get_policy().std.detach().mean().item()

        # Keep native metric namespaces and include keys introduced during the rollout.
        episode_lines = []
        ep_infos = locs['ep_infos']
        for key in sorted({key for ep_info in ep_infos for key in ep_info}):
            values = []
            for ep_info in ep_infos:
                if key not in ep_info:
                    continue
                value = torch.as_tensor(ep_info[key], device=self.device).detach().float().reshape(-1)
                if value.numel():
                    values.append(value)
            if not values:
                continue
            value = torch.cat(values).mean().item()
            if '/' in key:
                tag = key
                label = key
            else:
                if 'terrain' in key:
                    namespace = 'Terrain'
                elif 'max_command' in key:
                    namespace = 'CommandCurriculum'
                else:
                    namespace = 'Episode'
                tag = f'{namespace}/{key}'
                label = f'Mean episode {key}'
            self.writer.add_scalar(tag, value, it)
            episode_lines.append(f'{label + ":":>{pad}} {value:.4f}')

        for key, value in loss_dict.items():
            self.writer.add_scalar(f'Loss/{key}', value, it)
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, it)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std, it)
        self.writer.add_scalar('Perf/total_fps', fps, it)
        self.writer.add_scalar('Perf/collection_time', collection_time, it)
        self.writer.add_scalar('Perf/learning_time', learn_time, it)

        # Completed-episode returns remain separate from per-step reward metrics.
        reward_metrics = {}
        for name, (rewards, lengths) in reward_buffers.items():
            if not rewards:
                continue
            prefix = f'{name}_' if name else ''
            reward_metrics[f'mean_{prefix}reward'] = statistics.mean(rewards)
            reward_metrics[f'mean_{prefix}episode_length'] = statistics.mean(lengths)
        for key, value in reward_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, it)
            self.writer.add_scalar(f'Train/{key}/time', value, int(self.tot_time))

        title = f" Learning iteration {it}/{locs['tot_iter']} "
        lines = ['#' * width, f'\033[1m{title.center(width)}\033[0m', '']

        def row(label, value):
            lines.append(f'{label + ":":>{pad}} {value}')

        run_name = self.cfg.get('run_name')
        if run_name:
            row('Run name', run_name)
        row('Total steps', self.tot_timesteps)
        row('Steps per second', fps)
        row('Collection time', f'{collection_time:.3f}s')
        row('Learning time', f'{learn_time:.3f}s')
        for key, value in loss_dict.items():
            row(f'Mean {key} loss', f'{value:.4f}')
        for key, value in reward_metrics.items():
            row(key.replace('_', ' ').capitalize(), f'{value:.2f}')
        row('Mean action noise std', f'{mean_std:.2f}')
        lines.extend(episode_lines)

        completed_iterations = it - locs['start_iter'] + 1
        remaining_iterations = max(0, locs['tot_iter'] - it - 1)
        session_time = self.tot_time - locs['log_start_time']
        eta = session_time / completed_iterations * remaining_iterations
        lines.append('-' * width)
        row('Iteration time', f'{iteration_time:.2f}s')
        row('Time elapsed', str(timedelta(seconds=int(self.tot_time))))
        row('ETA', str(timedelta(seconds=int(eta))))
        print('\n'.join(lines) + '\n')

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.model.state_dict(),
            'optimizer1_state_dict': self.alg.optimizer1.state_dict(),
            'optimizer2_state_dict': self.alg.optimizer2.state_dict(),
            'iter': self.current_learning_iteration - 1,
            'next_iter': self.current_learning_iteration,
            'total_timesteps': self.tot_timesteps,
            'total_time': self.tot_time,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.alg.model.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer1.load_state_dict(loaded_dict['optimizer1_state_dict'])
            self.alg.optimizer2.load_state_dict(loaded_dict['optimizer2_state_dict'])
        # Legacy checkpoints store the next update index in "iter".
        self.current_learning_iteration = loaded_dict.get('next_iter', loaded_dict['iter'])
        self.tot_timesteps = loaded_dict.get('total_timesteps', 0)
        self.tot_time = loaded_dict.get('total_time', 0)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.model.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.model.to(device)
        def policy(observations):
            actor_obs = observations["actor"].flatten(1)
            policy_device = next(self.alg.model.parameters()).device
            return self.alg.model.act_inference(actor_obs.to(policy_device))

        return policy
