"""This module contains some base abstract classes for the models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Distribution
from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace

def weight_init(layer: torch.nn.modules):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(layer.weight.data, gain)
        if hasattr(layer.bias, 'data'): layer.bias.data.fill_(0.0)


def ln_activ(x: torch.Tensor, activ: Callable):
    x = F.layer_norm(x, (x.shape[-1],))
    return activ(x)


class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hdim: int, activ: str='elu'):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, output_dim)

        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, x: torch.Tensor):
        y = ln_activ(self.l1(x), self.activ)
        y = ln_activ(self.l2(y), self.activ)
        return self.l3(y)

class MRQActor(nn.Module):
    """An abstract class for actor.

    An actor approximates the policy function that maps observations to actions. Actor is
    parameterized by a neural network that takes observations as input, and outputs the mean and
    standard deviation of the action distribution.

    .. note::
        You can use this class to implement your own actor by inheriting it.

    Args:
        obs_space (OmnisafeSpace): observation space.
        act_space (OmnisafeSpace): action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        gumbel_tau: float=10, 
        zs_dim: int=512, 
        hdim: int=512, 
        activ: str='relu'
    ) -> None:
        """Initialize an instance of :class:`Actor`."""
        nn.Module.__init__(self)
        self._obs_space: OmnisafeSpace = obs_space
        self._act_space: OmnisafeSpace = act_space
        
        if isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1:
            self._obs_dim: int = obs_space.shape[0]
            self.discrete: bool = False
        elif isinstance(obs_space, spaces.Discrete):
            self._obs_dim: int = obs_space.n
            self.discrete: bool = True
        else:
            raise NotImplementedError
        if isinstance(act_space, spaces.Box) and len(act_space.shape) == 1:
            self._act_dim: int = act_space.shape[0]
        elif isinstance(act_space, spaces.Discrete):
            self._act_dim: int = act_space.n
        else:
            raise NotImplementedError
        
        
        self.policy = BaseMLP(zs_dim, self._act_dim, hdim, activ)
        self.activ = partial(F.gumbel_softmax, tau=gumbel_tau) if self.discrete else torch.tanh
        self._after_inference: bool = False

    def forward(self, zs: torch.Tensor):
        """Return the distribution of action.

        Args:
            obs (torch.Tensor): Observation from environments.
        """
        pre_activ = self.policy(zs)
        action = self.activ(pre_activ)
        return action, pre_activ

    def predict(
        self,
        zs: torch.Tensor
    ) -> torch.Tensor:
        r"""Predict deterministic or stochastic action based on observation.

        - ``deterministic`` = ``True`` or ``False``

        When training the actor, one important trick to avoid local minimum is to use stochastic
        actions, which can simply be achieved by sampling actions from the distribution (set
        ``deterministic=False``).

        When testing the actor, we want to know the actual action that the agent will take, so we
        should use deterministic actions (set ``deterministic=True``).

        .. math::

            L = -\underset{s \sim p(s)}{\mathbb{E}}[ \log p (a | s) A^R (s, a) ]

        where :math:`p (s)` is the distribution of observation, :math:`p (a | s)` is the
        distribution of action, and :math:`\log p (a | s)` is the log probability of action under
        the distribution., and :math:`A^R (s, a)` is the advantage function.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to predict deterministic action. Defaults to False.
        """
        action, _ = self.forward(zs)
        return action
    @property
    def noise(self) -> float:
        """Noise of the action."""
        return self._noise

    @noise.setter
    def noise(self, noise: float) -> None:
        """Set the action noise."""
        assert noise >= 0, 'Noise should be non-negative.'
        self._noise = noise
    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return self._noise