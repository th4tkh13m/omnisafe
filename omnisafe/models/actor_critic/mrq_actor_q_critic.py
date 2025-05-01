# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of ActorQCritic."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR

from omnisafe.models.actor.mrq_actor import MRQActor
from omnisafe.models.base import Actor, Critic
from omnisafe.models.critic.mrq_critic import MRQCritic
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config
from omnisafe.models.offline.mrq import Encoder


class MRQActorQCritic(nn.Module):
    """Class for ActorQCritic.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+---------------------------------------------------+
    | Model           | Description                                       |
    +=================+===================================================+
    | Actor           | Input is observation. Output is action.           |
    +-----------------+---------------------------------------------------+
    | Reward Q Critic | Input is obs-action pair. Output is reward value. |
    +-----------------+---------------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        target_actor (Actor): The target actor network.
        reward_critic (Critic): The critic network.
        target_reward_critic (Critic): The target critic network.
        actor_optimizer (Optimizer): The optimizer for the actor network.
        reward_critic_optimizer (Optimizer): The optimizer for the critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        actor: MRQActor,
        critic: MRQCritic,
        encoder: Encoder,
        actor_optimizer: optim.Optimizer,
        reward_critic_optimizer: optim.Optimizer,
        encoder_optimizer: optim.Optimizer,
        state_shape,
        discrete: bool = False,
        max_action: float = 1.0,
        device: str = 'cpu',
    ) -> None:
        """Initialize an instance of :class:`ActorQCritic`."""
        super().__init__()
        
        self.discrete: bool = discrete
        self.max_action: float = max_action
        self.state_shape = state_shape
        self._device: str = device

        self.actor: MRQActor = actor
        self.target_actor: MRQActor = deepcopy(self.actor)
        self.reward_critic: MRQCritic = critic
        self.target_reward_critic: MRQCritic = deepcopy(self.reward_critic)
        for param in self.target_reward_critic.parameters():
            param.requires_grad = False
        self.target_actor: Actor = deepcopy(
            self.actor,
        )
        for param in self.target_actor.parameters():
            param.requires_grad = False
        self.add_module('actor', self.actor)
        self.add_module('reward_critic', self.reward_critic)
        
        self.encoder : Encoder = encoder
        self.target_encoder: Encoder = deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.actor_optimizer: optim.Optimizer = actor_optimizer
        self.reward_critic_optimizer: optim.Optimizer = reward_critic_optimizer
        self.encoder_optimizer: optim.Optimizer = encoder_optimizer


    def step(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        with torch.no_grad():
            state = state.reshape(-1, *self.state_shape)
            zs = self.encoder.zs(state)
            action = self.actor.predict(zs)
            if not deterministic:
                action += torch.randn_like(action) * self.actor.noise
        if self.discrete:
            # for discrete action spaces, return a Python int
            return action.argmax().item()
        else:
            # for continuous actions, return a torch.Tensor
            # clamp into [-1,1], scale, then move to CPU
            return action.clamp(-1, 1).flatten() * self.max_action
                

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose the action based on the observation. used in training with gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        return self.step(obs, deterministic=deterministic)

    def update(self) -> None:
        """Update the target network
        """
        # for param, target_param in zip(
        #     self.reward_critic.parameters(),
        #     self.target_reward_critic.parameters(),
        # ):
        #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        # for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
        #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Update the target actor and critic networks using load_state_dict
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_reward_critic.load_state_dict(self.reward_critic.state_dict())
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        
