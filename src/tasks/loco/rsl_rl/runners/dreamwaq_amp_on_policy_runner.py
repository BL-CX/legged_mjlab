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

from ..algorithms import DreamWaQAMPPPO
from ..modules import DreamWaQActorCritic
from rsl_rl.env import VecEnv
from ..algorithms.amp_discriminator import AMPDiscriminator
from ..datasets.amp_motion_loader import AMPMotionLoader
from ..utils.utils import Normalizer

class DreamWaQAMPOnPolicyRunner:

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
        if history.ndim != 3 or history.shape[1] != 6:
            raise ValueError("DreamWaQ requires actor observations shaped [N,6,F]")
        self.num_one_step_obs = history.shape[-1]
        self.num_actor_obs = history.shape[1] * self.num_one_step_obs
        self.num_critic_obs = critic.shape[-1]
        
        # Initialize policy network
        actor_critic_class = eval(self.cfg["policy_class_name"])  # DreamWaQActorCritic
        actor_critic : DreamWaQActorCritic = actor_critic_class(
                                        self.num_actor_obs,
                                        self.num_critic_obs,
                                        self.num_one_step_obs,
                                        self.env.num_actions,
                                        **self.policy_cfg).to(self.device)
        amp_data = AMPMotionLoader(
            device, time_between_frames=self.env.unwrapped.step_dt, preload_transitions=True,
            num_preload_transitions=self.cfg['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"],
            reorder=self.cfg["reorder_motion_data"])
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            self.cfg['amp_reward_coef'],
            self.cfg['amp_discr_hidden_dims'], device
            ).to(self.device)
        min_std = (
            torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
            (torch.abs(self.env.dof_pos_limits[:, 1].to(self.device) - self.env.dof_pos_limits[:, 0].to(self.device))))
        
        # Initialize algorithm
        alg_class = eval(self.cfg["algorithm_class_name"])  # DreamWaQPPO
        self.alg : DreamWaQAMPPPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, min_std, device=self.device, **self.alg_cfg)
        
        # Training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Init storage and model
        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.num_actor_obs],
            [self.num_critic_obs], 
            [self.env.num_actions]
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        observations = self.env.get_observations()
        obs = observations["actor"].flatten(1).to(self.device)
        critic_obs = observations["critic"].to(self.device)
        amp_obs = observations["amp"].clone()
        amp_obs = amp_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs,amp_obs)
                    observations, rewards, dones, infos = self.env.step(actions)
                    obs = observations["actor"].flatten(1).to(self.device)
                    critic_obs = observations["critic"].to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    next_amp_obs = observations["amp"].to(self.device)
                    # Native MJLab auto-resets before observation computation.
                    # Approximate terminal AMP states with the pre-step states.
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    reset_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if len(reset_env_ids) > 0:
                        next_amp_obs_with_term[reset_env_ids] = amp_obs[reset_env_ids]
                    rewards,amp_rewards = self.alg.discriminator.predict_amp_reward(amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        elif 'log' in infos:
                            ep_infos.append(infos['log'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
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
            mean_value_loss, mean_surrogate_loss, mean_vae_loss,mean_vel_estimation_loss, \
            mean_recon_loss, mean_kl_loss,mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, mean_entropy_loss= self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it + 1
            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()
        
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            self.writer.flush()

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if key not in ep_info:
                        continue
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if 'terrain' in key:
                    self.writer.add_scalar('Terrain/' + key, value, locs['it'])
                elif 'max_command' in key:
                    self.writer.add_scalar('CommandCurriculum/' + key, value, locs['it'])
                else:
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/entropy', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/vae_function', locs['mean_vae_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_kl_loss', locs['mean_kl_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_vel_estimation_loss', locs['mean_vel_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_recon_loss', locs['mean_recon_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                          f"""{'vae function loss:':>{pad}} {locs['mean_vae_loss']:.4f}\n"""
                          f"""{'Vel Estimation loss:':>{pad}} {locs['mean_vel_estimation_loss']:.4f}\n"""
                          f"""{'Recon loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                          f"""{'VAE KL Divergence Loss:':>{pad}} {locs['mean_kl_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                          f"""{'vae function loss:':>{pad}} {locs['mean_vae_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'vae_optimizer_state_dict': self.alg.vae_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.vae_optimizer.load_state_dict(loaded_dict['vae_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
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
