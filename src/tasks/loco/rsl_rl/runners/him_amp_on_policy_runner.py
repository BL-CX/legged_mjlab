
import time
import os
from collections import deque
from datetime import timedelta
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from ..algorithms.him_amp_ppo import HIMAMPPPO
from ..modules.him_amp_actor_critic import HIMAMPActorCritic
from rsl_rl.env import VecEnv
from ..utils.utils import Normalizer
from ..algorithms.amp_discriminator import AMPDiscriminator
from ..datasets.amp_motion_loader import AMPMotionLoader

class HIMAMPOnPolicyRunner:

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
        observations = self.env.get_observations()
        history = observations["actor"]
        critic = observations["critic"]
        amp_obs = observations["amp"]
        if history.ndim != 3 or critic.ndim != 2 or amp_obs.ndim != 2:
            raise ValueError("AMP requires actor [N,H,F], critic [N,C], amp [N,D]")
        self.num_one_step_obs = history.shape[-1]
        self.num_actor_obs = history.shape[1] * history.shape[2]
        self.num_critic_obs = critic.shape[-1]
        actor_critic_class = eval(self.cfg["policy_class_name"]) # HIMAMPActorCritic
        actor_critic: HIMAMPActorCritic = actor_critic_class( self.num_actor_obs,
                                                        self.num_critic_obs,
                                                        self.num_one_step_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)

        amp_data = AMPMotionLoader(
            device, time_between_frames=self.env.unwrapped.step_dt, preload_transitions=True,
            num_preload_transitions=self.cfg["amp_num_preload_transitions"],
            motion_files=self.cfg["amp_motion_files"],
            reorder=self.cfg.get("reorder_motion_data", False))
        if amp_obs.shape[-1] != amp_data.observation_dim:
            raise ValueError(
                "AMP observation dimension mismatch: environment provides "
                f"{amp_obs.shape[-1]}, motion data provides "
                f"{amp_data.observation_dim}. Check AMP joint/foot ordering."
            )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            self.cfg["amp_reward_coef"],
            self.cfg["amp_discr_hidden_dims"], device,
            amp_task_reward_lerp=self.cfg["amp_task_reward_lerp"],
            ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"]) # HIMAMPPPO
        joint_limits = self.env.dof_pos_limits.to(self.device)
        min_std = (
            torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
            (torch.abs(joint_limits[:, 1] - joint_limits[:, 0])))
        self.alg: HIMAMPPPO = alg_class(actor_critic,discriminator, amp_data, amp_normalizer, device=self.device, min_std=min_std, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.num_actor_obs], [self.num_critic_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        # Index of the next policy update.
        self.current_learning_iteration = 0

        # The official wrapper has already reset the environment.

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        observations = self.env.get_observations()
        obs = observations["actor"].flatten(1).to(self.device)
        critic_obs = observations["critic"].to(self.device)
        amp_obs = observations["amp"].detach().clone().to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        style_rewbuffer = deque(maxlen=100)
        task_rewbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        task_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        style_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        log_start_time = self.tot_time
        for it in range(start_iter, tot_iter):
            start = time.time()
            iteration_reward_sum = torch.zeros((), device=self.device)
            iteration_task_reward_sum = torch.zeros((), device=self.device)
            iteration_style_reward_sum = torch.zeros((), device=self.device)
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    previous_critic_obs = critic_obs.detach().clone()
                    actions = self.alg.act(obs, previous_critic_obs,amp_obs)
                    observations, rewards, dones, infos = self.env.step(actions)
                    obs = observations["actor"].flatten(1).to(self.device)
                    critic_obs = observations["critic"].to(self.device)
                    next_amp_obs = observations["amp"].to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    # MJLab returns post-reset observations for done rows. As in
                    # AMP_mjlab, approximate their terminal targets with the
                    # observations from before this action.
                    next_amp_obs_with_term = next_amp_obs.detach().clone()
                    next_critic_obs = critic_obs.detach().clone()
                    termination_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if len(termination_ids) > 0:
                        next_amp_obs_with_term[termination_ids] = amp_obs[termination_ids]
                        next_critic_obs[termination_ids] = previous_critic_obs[termination_ids]
                    self.task_rewards = rewards
                    rewards,self.amp_rewards = self.alg.discriminator.predict_amp_reward(amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)
                    iteration_reward_sum += rewards.mean()
                    iteration_task_reward_sum += self.task_rewards.mean()
                    iteration_style_reward_sum += self.amp_rewards.mean()
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs,next_amp_obs_with_term)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        elif 'log' in infos:
                            ep_infos.append(infos['log'])
                        cur_reward_sum += rewards
                        task_reward_sum += self.task_rewards
                        style_reward_sum += self.amp_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        style_rewbuffer.extend(style_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        task_rewbuffer.extend(task_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        style_reward_sum[new_ids] = 0
                        task_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                mean_reward_per_step = (iteration_reward_sum / self.num_steps_per_env).item()
                mean_task_reward_per_step = (iteration_task_reward_sum / self.num_steps_per_env).item()
                mean_style_reward_per_step = (iteration_style_reward_sum / self.num_steps_per_env).item()

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss,\
                mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it + 1
            if self.log_dir is not None:
                self.log(locals())
            if self.log_dir is not None and it % self.save_interval == 0:
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
            'estimation': locs['mean_estimation_loss'],
            'swap': locs['mean_swap_loss'],
            'amp': locs['mean_amp_loss'],
            'amp_grad_penalty': locs['mean_grad_pen_loss'],
        }
        train_metrics = {
            'mean_reward_per_step': locs['mean_reward_per_step'],
            'mean_task_reward_per_step': locs['mean_task_reward_per_step'],
            'mean_style_reward_per_step': locs['mean_style_reward_per_step'],
        }
        if locs['task_rewbuffer']:
            train_metrics['mean_task_reward'] = statistics.mean(locs['task_rewbuffer'])
            train_metrics['mean_style_reward'] = statistics.mean(locs['style_rewbuffer'])
        amp_metrics = {
            'mean_policy_pred': locs['mean_policy_pred'],
            'mean_expert_pred': locs['mean_expert_pred'],
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
        if locs['rewbuffer']:
            reward_metrics['mean_reward'] = statistics.mean(locs['rewbuffer'])
            reward_metrics['mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        for key, value in reward_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, it)
            self.writer.add_scalar(f'Train/{key}/time', value, int(self.tot_time))
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, it)
        for key, value in amp_metrics.items():
            self.writer.add_scalar(f'AMP/{key}', value, it)

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
        for key, value in amp_metrics.items():
            row('AMP ' + key.replace('_', ' '), f'{value:.4f}')
        for key, value in reward_metrics.items():
            row(key.replace('_', ' ').capitalize(), f'{value:.2f}')
        for key, value in train_metrics.items():
            row(key.replace('_', ' ').capitalize(), f'{value:.4f}')
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
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration - 1,
            'next_iter': self.current_learning_iteration,
            'total_timesteps': self.tot_timesteps,
            'total_time': self.tot_time,
            'infos': infos,
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        # Legacy checkpoints store the next update index in "iter".
        self.current_learning_iteration = loaded_dict.get('next_iter', loaded_dict['iter'])
        self.tot_timesteps = loaded_dict.get('total_timesteps', 0)
        self.tot_time = loaded_dict.get('total_time', 0)
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
