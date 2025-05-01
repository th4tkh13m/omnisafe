"""Implementation of the Model-based Representation Q Learning algorithm."""

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import dataclasses
import time
import copy
import torch.nn.functional as F
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.models.actor.mrq_actor import MRQActor
from omnisafe.models.critic.mrq_critic import MRQCritic
from omnisafe.models.actor_critic.mrq_actor_q_critic import MRQActorQCritic
from omnisafe.models.offline.mrq import Encoder
from gymnasium import spaces
from omnisafe.common.buffer.mrq_buffer import ReplayBuffer
from omnisafe.adapter.mrq_adapter import MRQAdapter
from omnisafe.common.logger import Logger
import numpy as np

from typing import Any



@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class MRQ(BaseAlgo):
    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OffPolicyAdapter` to adapt the environment to this
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
            AssertionError: If the total number of steps is not divisible by the number of steps per
                epoch.
        """
        self._env: MRQAdapter = MRQAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), 'The number of steps per epoch is not divisible by the number of environments.'

        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), 'The total number of steps is not divisible by the number of steps per epoch.'
        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )

        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle
        assert (
            self._steps_per_epoch % self._update_cycle == 0
        ), 'The number of steps per epoch is not divisible by the number of steps per sample.'
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    def _init_model(self) -> None:
        """Initialize the model.
        """
        self.name = 'MR.Q'

        # self.hp = Hyperparameters(**hp)
        # utils.set_instance_vars(self.hp, self)
  
        self.history = 1
        self.exploration_noise = self._cfgs.algo_cfgs.exploration_noise
        self.noise_clip = self._cfgs.algo_cfgs.noise_clip
        self.target_policy_noise = self._cfgs.algo_cfgs.target_policy_noise
        if isinstance(self._env.observation_space, spaces.Discrete):
            # Scale action noise since discrete actions are [0,1] and continuous actions are [-1,1].
            self.exploration_noise *= 0.5
            self.noise_clip *= 0.5
            self.target_policy_noise *= 0.5
            self.discrete = True
        elif isinstance(self._env.observation_space, spaces.Box):
            self.discrete = False
            
        
        
        self.obs_shape = self._env.observation_space.shape # Size of individual frames.
        self.obs_dtype = torch.float

        # Size of state given to network
        self.state_shape = [self.obs_shape[0] * self.history]
        
        self.max_action = 1 if self.discrete else float(self._env.action_space.high[0])

        

        self.encoder = Encoder(self._env.observation_space, self._env.action_space,
            self._cfgs.algo_cfgs.num_bins,
            self._cfgs.algo_cfgs.zs_dim, 
            self._cfgs.algo_cfgs.za_dim, 
            self._cfgs.algo_cfgs.zsa_dim,
            self._cfgs.algo_cfgs.enc_hdim, 
            self._cfgs.algo_cfgs.enc_activ).to(self._device)
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), 
                                                   lr=self._cfgs.algo_cfgs.enc_lr, 
                                                   weight_decay=self._cfgs.algo_cfgs.enc_wd)

        self.policy = MRQActor(self._env.observation_space, 
                               self._env.action_space, 
                               self._cfgs.algo_cfgs.gumbel_tau, 
                               self._cfgs.algo_cfgs.zs_dim,
                                self._cfgs.algo_cfgs.policy_hdim, 
                                self._cfgs.algo_cfgs.policy_activ).to(self._device)
        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), 
                                                  lr=self._cfgs.algo_cfgs.policy_lr, 
                                                  weight_decay=self._cfgs.algo_cfgs.policy_wd)

        self.value = MRQCritic(self._cfgs.algo_cfgs.zsa_dim, 
                               self._cfgs.algo_cfgs.value_hdim, 
                               self._cfgs.algo_cfgs.value_activ).to(self._device)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), 
                                                 lr=self._cfgs.algo_cfgs.value_lr, 
                                                 weight_decay=self._cfgs.algo_cfgs.value_wd)

        # Used by reward prediction
        self.two_hot = TwoHot(self._device, 
                              self._cfgs.algo_cfgs.lower, 
                              self._cfgs.algo_cfgs.upper, 
                              self._cfgs.algo_cfgs.num_bins)


        
        
        self._actor_critic = MRQActorQCritic(
            actor=self.policy,
            critic=self.value,
            encoder=self.encoder,
            actor_optimizer=self.policy_optimizer,
            reward_critic_optimizer=self.value_optimizer,
            encoder_optimizer=self.encoder_optimizer,
            state_shape=self.state_shape,
            discrete=self.discrete,
            max_action=self.max_action,
            device=self._device,
        )
        self.pixel_obs = False
        
    
    
    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self.replay_buffer = ReplayBuffer(
            self._env.observation_space,
            self._env.action_space, 
            self.max_action, 
            self._device,
            self.history,
            max(self._cfgs.algo_cfgs.enc_horizon, self._cfgs.algo_cfgs.Q_horizon),
            self._cfgs.algo_cfgs.buffer_size,
            self._cfgs.algo_cfgs.batch_size,
            self._cfgs.algo_cfgs.prioritized,
            initial_priority=self._cfgs.algo_cfgs.min_priority)
        self.state_shape = self.replay_buffer.state_shape # This includes history, horizon, channels, etc.

        # Tracked values
        self.reward_scale, self.target_reward_scale = 1, 0
        self.training_steps = 0
        
    def _init_log(self) -> None:
        """Log info about epoch.

        +-------------------------+----------------------------------------------------------------------+
        | Things to log           | Description                                                          |
        +=========================+======================================================================+
        | Train/Epoch             | Current epoch.                                                       |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpCost          | Average cost of the epoch.                                           |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpRet           | Average return of the epoch.                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpLen           | Average length of the epoch.                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpCost      | Average cost of the evaluate epoch.                                  |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpRet       | Average return of the evaluate epoch.                                |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpLen       | Average length of the evaluate epoch.                                |
        +-------------------------+----------------------------------------------------------------------+
        | Value/reward_critic     | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-------------------------+----------------------------------------------------------------------+
        | Values/cost_critic      | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi            | Loss of the policy network.                                          |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_reward_critic | Loss of the reward critic.                                           |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic   | Loss of the cost critic network.                                     |
        +-------------------------+----------------------------------------------------------------------+
        | Train/LR                | Learning rate of the policy network.                                 |
        +-------------------------+----------------------------------------------------------------------+
        | Misc/Seed               | Seed of the experiment.                                              |
        +-------------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps      | Total steps of the experiment.                                       |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Total              | Total time.                                                          |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Rollout            | Rollout time.                                                        |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Update             | Update time.                                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Evaluate           | Evaluate time.                                                       |
        +-------------------------+----------------------------------------------------------------------+
        | FPS                     | Frames per second of the epoch.                                      |
        +-------------------------+----------------------------------------------------------------------+
        """
        self._logger: Logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['actor_critic'] = self._actor_critic

        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key(
            'Metrics/EpRet',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpCost',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpLen',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

        if self._cfgs.train_cfgs.eval_episodes > 0:
            self._logger.register_key(
                'Metrics/TestEpRet',
                window_length=self._cfgs.logger_cfgs.window_lens,
            )
            self._logger.register_key(
                'Metrics/TestEpCost',
                window_length=self._cfgs.logger_cfgs.window_lens,
            )
            self._logger.register_key(
                'Metrics/TestEpLen',
                window_length=self._cfgs.logger_cfgs.window_lens,
            )

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/LR')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward_critic')
        
        self._logger.register_key('Loss/Loss_encoder', delta=True)

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost_critic')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Evaluate')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')
        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self.logger.register_key(env_spec_key)
            
            
    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: average episode return in final epoch.
            ep_cost: average episode cost in final epoch.
            ep_len: average episode length in final epoch.
        """
        self._logger.log('INFO: Start training')
        
        
        state, _ = self._env.reset()
        start_time = time.time()
        step = 0
        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch,
            ):

                rollout_start = time.time()
                # set noise for exploration
                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = self.exploration_noise

                action = self._actor_critic.step(state)
                
                next_state, reward, cost, terminated, truncated, info = self._env.step(action)
                self.replay_buffer.add(
                    state,
                    action,
                    next_state,
                    reward,
                    terminated,
                    truncated
                )
                self._env._log_value(reward=reward, cost=cost, info=info)
                real_next_obs = next_state.clone()
                for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                    if done:
                        if 'final_observation' in info:
                            real_next_obs[idx] = info['final_observation'][idx]
                        self._env._log_metrics(self._logger, idx)
                        self._env._reset_log(idx)
                state = next_state
                rollout_time += time.time() - rollout_start

                # update parameters
                update_start = time.time()
                self._train(step)

                # if we haven't updated the network, log 0 for the loss
                # else:
                #     self._log_when_not_update()
                
                
                
                update_time += time.time() - update_start
                
                

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            eval_time = time.time() - eval_start

            self._logger.store({'Time/Update': update_time})
            self._logger.store({'Time/Rollout': rollout_time})
            self._logger.store({'Time/Evaluate': eval_time})

            self._logger.store(
                {
                    'TotalEnvSteps': step + 1,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len
    
    def _train(self, step) -> None:
        """Train the algorithm.

        -  Update the actor and critic networks.
        -  Update the encoder network.
        """
        if step % self._cfgs.algo_cfgs.target_update_freq == 0:
            self._update()
            self._update_encoder()
        buffer_state, buffer_action, buffer_next_state, buffer_reward, buffer_not_done = self.replay_buffer.sample(self._cfgs.algo_cfgs.Q_horizon, include_intermediate=False)
        buffer_state, buffer_next_state = maybe_augment_state(buffer_state, buffer_next_state, self.pixel_obs, self._cfgs.algo_cfgs.pixel_augs)
        buffer_reward, buffer_term_discount = multi_step_reward(buffer_reward, buffer_not_done, self._cfgs.algo_cfgs.discount)
        
        Q, Q_target = self.train_rl(buffer_state, buffer_action, buffer_next_state, buffer_reward, buffer_term_discount,
            self.reward_scale, self.target_reward_scale)

        if self._cfgs.algo_cfgs.prioritized:
            priority = (Q - Q_target.expand(-1,2)).abs().max(1).values
            priority = priority.clamp(min=self._cfgs.algo_cfgs.min_priority).pow(self._cfgs.algo_cfgs.alpha)
            self.replay_buffer.update_priority(priority)
    
    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. note::

            +----------+---------------------------------------+
            | obs      | ``observaion`` stored in buffer.      |
            +==========+=======================================+
            | act      | ``action`` stored in buffer.          |
            +----------+---------------------------------------+
            | reward   | ``reward`` stored in buffer.          |
            +----------+---------------------------------------+
            | cost     | ``cost`` stored in buffer.            |
            +----------+---------------------------------------+
            | next_obs | ``next observaion`` stored in buffer. |
            +----------+---------------------------------------+
            | done     | ``terminated`` stored in buffer.      |
            +----------+---------------------------------------+

        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the ``update_iters`` times.
        """
        self._actor_critic.update()

            # self._update_reward_critic(obs, act, reward, done, next_obs)
            # if self._cfgs.algo_cfgs.use_cost:
            #     self._update_cost_critic(obs, act, cost, done, next_obs)

            # if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
            #     self._update_actor(obs)
            #     self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)
    
    def _update_encoder(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. note::

            +----------+---------------------------------------+
            | obs      | ``observaion`` stored in buffer.      |
            +==========+=======================================+
            | act      | ``action`` stored in buffer.          |
            +----------+---------------------------------------+
            | reward   | ``reward`` stored in buffer.          |
            +----------+---------------------------------------+
            | cost     | ``cost`` stored in buffer.            |
            +----------+---------------------------------------+
            | next_obs | ``next observaion`` stored in buffer. |
            +----------+---------------------------------------+
            | done     | ``terminated`` stored in buffer.      |
            +----------+---------------------------------------+

        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the ``update_iters`` times.
        """
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            state, action, next_state, reward, not_done = self.replay_buffer.sample(self._cfgs.algo_cfgs.enc_horizon, include_intermediate=True)
            self._update_count += 1
            
            state, next_state = maybe_augment_state(state, next_state, self.pixel_obs, self._cfgs.algo_cfgs.pixel_augs)
            self.train_encoder(state, action, next_state, reward, not_done, self.replay_buffer.env_terminates)
            

            # self._update_reward_critic(obs, act, reward, done, next_obs)
            # if self._cfgs.algo_cfgs.use_cost:
            #     self._update_cost_critic(obs, act, cost, done, next_obs)

            # if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
            #     self._update_actor(obs)
            #     self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)
    
    def train_encoder(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
        reward: torch.Tensor, not_done: torch.Tensor, env_terminates: bool):
        with torch.no_grad():
            encoder_target = self._actor_critic.target_encoder.zs(
                next_state.reshape(-1,*self.state_shape) # Combine batch and horizon
            ).reshape(state.shape[0],-1,self._cfgs.algo_cfgs.zs_dim) # Separate batch and horizon

        pred_zs = self._actor_critic.encoder.zs(state[:,0])
        prev_not_done = 1 # In subtrajectories with termination, mask out losses after termination.
        encoder_loss = 0 # Loss is accumluated over enc_horizon.

        for i in range(self._cfgs.algo_cfgs.enc_horizon):
            pred_d, pred_zs, pred_r = self._actor_critic.encoder.model_all(pred_zs, action[:,i])

            # Mask out states past termination.
            dyn_loss = masked_mse(pred_zs, encoder_target[:,i], prev_not_done)
            reward_loss = (self.two_hot.cross_entropy_loss(pred_r, reward[:,i]) * prev_not_done).mean()
            done_loss = masked_mse(pred_d, 1. - not_done[:,i].reshape(-1,1), prev_not_done) if env_terminates else 0

            encoder_loss = encoder_loss + self._cfgs.algo_cfgs.dyn_weight * dyn_loss + self._cfgs.algo_cfgs.reward_weight * reward_loss + self._cfgs.algo_cfgs.done_weight * done_loss
            prev_not_done = not_done[:,i].reshape(-1,1) * prev_not_done # Adjust termination mask.

        self._actor_critic.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self._actor_critic.encoder_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_encoder': encoder_loss.item(),
            },
        )


    def train_rl(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
        reward: torch.Tensor, term_discount: torch.Tensor, reward_scale: float, target_reward_scale: float):
        with torch.no_grad():
            next_zs = self._actor_critic.target_encoder.zs(next_state)

            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = realign(self._actor_critic.target_actor.predict(next_zs) + noise, self.discrete) # Clips to (-1,1) OR one_hot of argmax.

            next_zsa = self._actor_critic.target_encoder(next_zs, next_action)
            Q_target = self._actor_critic.target_reward_critic(next_zsa).min(1,keepdim=True).values
            Q_target = (reward + term_discount * Q_target * target_reward_scale)/reward_scale

            zs = self._actor_critic.encoder.zs(state)
            zsa = self._actor_critic.encoder(zs, action)

        Q = self._actor_critic.reward_critic(zsa)
        value_loss = F.smooth_l1_loss(Q, Q_target.expand(-1,2))

        self._actor_critic.reward_critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor_critic.reward_critic.parameters(), self._cfgs.algo_cfgs.value_grad_clip)
        self._actor_critic.reward_critic_optimizer.step()

        policy_action, pre_activ = self._actor_critic.actor(zs)
        zsa = self._actor_critic.encoder(zs, policy_action)
        Q_policy = self._actor_critic.reward_critic(zsa)
        policy_loss = -Q_policy.mean() + self._cfgs.algo_cfgs.pre_activ_weight * pre_activ.pow(2).mean()

        self._actor_critic.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_pi': policy_loss.item(),
                'Loss/Loss_reward_critic': value_loss.item(),
                'Value/reward_critic': Q.mean().item(),
            },
        )

        return Q, Q_target


    def save(self, save_folder: str):
        # Save models/optimizers
        models = [
            'encoder', 'encoder_target', 'encoder_optimizer',
            'policy', 'policy_target', 'policy_optimizer',
            'value', 'value_target', 'value_optimizer'
        ]
        for k in models: torch.save(self.__dict__[k].state_dict(), f'{save_folder}/{k}.pt')

        # Save variables
        vars = ['hp', 'reward_scale', 'target_reward_scale', 'training_steps']
        var_dict = {k: self.__dict__[k] for k in vars}
        np.save(f'{save_folder}/agent_var.npy', var_dict)

        self.replay_buffer.save(save_folder)


    def load(self, save_folder: str):
        # Load models/optimizers.
        models = [
            'encoder', 'encoder_target', 'encoder_optimizer',
            'policy', 'policy_target', 'policy_optimizer',
            'value', 'value_target', 'value_optimizer'
        ]
        for k in models: self.__dict__[k].load_state_dict(torch.load(f'{save_folder}/{k}.pt', weights_only=True))

        # Load variables.
        var_dict = np.load(f'{save_folder}/agent_var.npy', allow_pickle=True).item()
        for k, v in var_dict.items(): self.__dict__[k] = v

        self.replay_buffer.load(save_folder)
    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        self._logger.store(
            {
                'Loss/Loss_reward_critic': 0.0,
                'Loss/Loss_pi': 0.0,
                'Value/reward_critic': 0.0,
            },
        )
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.store(
                {
                    'Loss/Loss_cost_critic': 0.0,
                    'Value/cost_critic': 0.0,
                },
            )

class TwoHot:
    def __init__(self, device: torch.device, lower: float=-10, upper: float=10, num_bins: int=101):
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1) # symexp
        self.num_bins = num_bins


    def transform(self, x: torch.Tensor):
        diff = x - self.bins.reshape(1,-1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind+1).clamp(0, self.num_bins-1)]
        weight = (x - lower)/(upper - lower)

        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind+1).clamp(0, self.num_bins), weight)
        return two_hot


    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)


    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.transform(target)
        return -(target * pred).sum(-1, keepdim=True)


def realign(x, discrete: bool):
    return F.one_hot(x.argmax(1), x.shape[1]).float() if discrete else x.clamp(-1,1)


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    return (F.mse_loss(x, y, reduction='none') * mask).mean()


def multi_step_reward(reward: torch.Tensor, not_done: torch.Tensor, discount: float):
    ms_reward = 0
    scale = 1
    for i in range(reward.shape[1]):
        ms_reward += scale * reward[:,i]
        scale *= discount * not_done[:,i]
    
    return ms_reward, scale


def maybe_augment_state(state: torch.Tensor, next_state: torch.Tensor, pixel_obs: bool, use_augs: bool):
    if pixel_obs and use_augs:
        if len(state.shape) != 5: state = state.unsqueeze(1)
        batch_size, horizon, history, height, width = state.shape

        # Group states before augmenting.
        both_state = torch.concatenate([state.reshape(-1, history, height, width), next_state.reshape(-1, history, height, width)], 0)
        both_state = shift_aug(both_state)

        state, next_state = torch.chunk(both_state, 2, 0)
        state = state.reshape(batch_size, horizon, history, height, width)
        next_state = next_state.reshape(batch_size, horizon, history, height, width)

        if horizon == 1:
            state = state.squeeze(1)
            next_state = next_state.squeeze(1)
    return state, next_state


# Random shift.
def shift_aug(image: torch.Tensor, pad: int=4):
    batch_size, _, height, width = image.size()
    image = F.pad(image, (pad, pad, pad, pad), 'replicate')
    eps = 1.0 / (height + 2 * pad)

    arange = torch.linspace(-1.0 + eps, 1.0 - eps, height + 2 * pad, device=image.device, dtype=torch.float)[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)

    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(batch_size, 1, 1, 2), device=image.device, dtype=torch.float)
    shift *= 2.0 / (height + 2 * pad)
    return F.grid_sample(image, base_grid + shift, padding_mode='zeros', align_corners=False)
        