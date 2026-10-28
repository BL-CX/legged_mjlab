
import time
import os
from collections import deque
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

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
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
                        elif 'log' in infos and dones.any():
                            # Isaac Lab 2.x publishes per-episode reward terms
                            # under "log" instead of the legacy "episode" key.
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
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

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
                if infotensor.numel() > 0:
                    value = torch.mean(infotensor)
                    if "/" in key:
                        self.writer.add_scalar(key, value, locs['it'])
                        ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar('Episode/' + key, value, locs['it'])
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/Swap Loss', locs['mean_swap_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        self.writer.add_scalar('Train/mean_reward_per_step', locs['mean_reward_per_step'], locs['it'])
        self.writer.add_scalar('Train/mean_task_reward_per_step', locs['mean_task_reward_per_step'], locs['it'])
        self.writer.add_scalar('Train/mean_style_reward_per_step', locs['mean_style_reward_per_step'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_style_reward', statistics.mean(locs['style_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_task_reward', statistics.mean(locs['task_rewbuffer']), locs['it'])
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
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean task reward:':>{pad}} {statistics.mean(locs['task_rewbuffer']):.2f}\n"""
                          f"""{'Mean style reward:':>{pad}} {statistics.mean(locs['style_rewbuffer']):.2f}\n"""
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
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
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
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
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
