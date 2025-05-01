
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

class MRQCritic(nn.Module):
    """An abstract class for critic.

    A critic approximates the value function that maps observations to values. Critic is
    parameterized by a neural network that takes observations as input, (Q critic also takes actions
    as input) and outputs the value estimated.

    .. note::
        OmniSafe provides two types of critic:
        Q critic (Input = ``observation`` + ``action`` , Output = ``value``),
        and V critic (Input = ``observation`` , Output = ``value``).
        You can also use this class to implement your own actor by inheriting it.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
        use_obs_encoder (bool, optional): Whether to use observation encoder, only used in q critic.
            Defaults to False.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        zsa_dim: int=512, 
        hdim: int=512, 
        activ: str='elu'
    ) -> None:
        """Initialize an instance of :class:`Critic`."""
        nn.Module.__init__(self)

        class ValueNetwork(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hdim: int=512, activ: str='elu'):
                super().__init__()
                self.q1 = BaseMLP(input_dim, hdim, hdim, activ)
                self.q2 = nn.Linear(hdim, output_dim)

                self.activ = getattr(F, activ)
                self.apply(weight_init)

            def forward(self, zsa: torch.Tensor):
                zsa = ln_activ(self.q1(zsa), self.activ)
                return self.q2(zsa)

        self.q1 = ValueNetwork(zsa_dim, 1, hdim, activ)
        self.q2 = ValueNetwork(zsa_dim, 1, hdim, activ)


    def forward(self, zsa: torch.Tensor):
        return torch.cat([self.q1(zsa), self.q2(zsa)], 1)
    

