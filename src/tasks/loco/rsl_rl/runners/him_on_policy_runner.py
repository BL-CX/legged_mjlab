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
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from ..algorithms import HIMPPO
from ..modules import HIMActorCritic
from rsl_rl.env import VecEnv


class HIMOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        """Initialize HimLoco on-policy runner.

        Args:
            env: Vectorized environment.
            train_cfg: Training configuration dictionary.
            log_dir: Directory for logging.
            device: Device to run on.
        """
        # Store configuration
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Derive dimensions from named observations, not ambiguous env attributes.
        observations = self.env.get_observations()
        history = observations["actor"]
        critic = observations["critic"]
        if history.ndim != 3 or critic.ndim != 2:
            raise ValueError("HIM requires actor [N,H,F] and critic [N,C]")
        self.num_one_step_obs = history.shape[-1]
        self.num_actor_obs = history.shape[1] * history.shape[2]
        self.num_critic_obs = critic.shape[-1]

        # Initialize policy network
        actor_critic_class = eval(self.cfg["policy_class_name"])  # HIMActorCritic
        actor_critic: HIMActorCritic = actor_critic_class(
            self.num_actor_obs,
            self.num_critic_obs,
            self.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)

        # Initialize algorithm
        alg_class = eval(self.cfg["algorithm_class_name"])  # HIMPPO
        self.alg: HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        # Training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Initialize storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.num_actor_obs],
            [self.num_critic_obs],
            [self.env.num_actions]
        )

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.logger_type = self.cfg["logger"].lower()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Train the policy using HimLoco PPO algorithm.

        Args:
            num_learning_iterations: Number of policy updates.
            init_at_random_ep_len: Whether to randomize initial episode lengths.
        """
        # Initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        observations = self.env.get_observations()
        obs = observations["actor"].flatten(1).to(self.device) #Flatten native actor histories, preserving MJLab's oldest-first order.
        critic_obs = observations["critic"].to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    previous_critic_obs = critic_obs.detach().clone()
                    actions = self.alg.act(obs, previous_critic_obs)
                    observations, rewards, dones, infos = self.env.step(actions)
                    obs = observations["actor"].flatten(1).to(self.device)
                    critic_obs = observations["critic"].to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    # Native MJLab returns post-reset observations. Approximate
                    # terminal targets with the corresponding pre-step critic.
                    next_critic_obs = critic_obs.detach().clone()
                    termination_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if len(termination_ids) > 0:
                        next_critic_obs[termination_ids] = previous_critic_obs[termination_ids]
                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        elif 'log' in infos:
                            ep_infos.append(infos['log'])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it + 1
            # log info
            if self.log_dir is not None:
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            # Clear episode infos
            ep_infos.clear()

        # Save the final model after training
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        """Log training information to console and TensorBoard.

        Enhanced version compatible with RSL-RL style logging, supporting:
        - Dynamic episode info processing
        - Loss dictionary iteration
        - Detailed performance metrics
        """
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        # -- Episode info
        ep_string = ""
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # skip missing keys
                    if key not in ep_info:
                        continue
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))

                if infotensor.numel() > 0:
                    value = torch.mean(infotensor)
                    # log to tensorboard with proper namespace
                    if "/" in key:
                        self.writer.add_scalar(key, value, locs['it'])
                        ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar('Episode/' + key, value, locs['it'])
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # -- Policy metrics
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # -- Losses
        loss_dict = {
            'value_function': locs['mean_value_loss'],
            'surrogate': locs['mean_surrogate_loss'],
            'estimation': locs['mean_estimation_loss'],
            'swap': locs['mean_swap_loss'],
        }
        for key, value in loss_dict.items():
            self.writer.add_scalar(f'Loss/{key}', value, locs['it'])

        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])

        # -- Performance
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection_time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        # -- Training metrics
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
                self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        # -- Console output
        str_header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # Add losses
            for key, value in loss_dict.items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # Add rewards
            log_string += (
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in loss_dict.items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        # Add episode info
        log_string += ep_string

        # Add summary
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                    locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        """Save model checkpoint.

        Args:
            path: Path to save the model.
            infos: Additional information to save.
        """
        saved_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if hasattr(self, 'logger_type') and self.logger_type in ["neptune", "wandb"]:
            if hasattr(self.writer, 'save_model'):
                self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        def policy(observations):
            actor_obs = observations["actor"].flatten(1)
            policy_device = next(self.alg.actor_critic.parameters()).device
            return self.alg.actor_critic.act_inference(actor_obs.to(policy_device))

        return policy
