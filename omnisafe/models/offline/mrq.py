from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnisafe.typing import OmnisafeSpace
from gymnasium import spaces


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


class Encoder(nn.Module):
    def __init__(self, 
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        num_bins: int=65,
        zs_dim: int=512,
        za_dim: int=256,
        zsa_dim: int=512,
        hdim: int=512,
        activ: str='elu'):
        super().__init__()
        
        # if isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1:
        #     self.state_dim = obs_space.shape[0]
        #     self.zs = self.cnn_zs
        #     self.zs_cnn1 = nn.Conv2d(self.state_dim, 32, 3, stride=2)
        #     self.zs_cnn2 = nn.Conv2d(32, 32, 3, stride=2)
        #     self.zs_cnn3 = nn.Conv2d(32, 32, 3, stride=2)
        #     self.zs_cnn4 = nn.Conv2d(32, 32, 3, stride=1)
        #     self.zs_lin = nn.Linear(1568, zs_dim)
        # elif isinstance(obs_space, spaces.Discrete):
        if isinstance(obs_space, spaces.Discrete):
            self.state_dim = obs_space.n
        elif isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1:
            self.state_dim = obs_space.shape[0]
        self.zs = self.mlp_zs
        self.zs_mlp = BaseMLP(self.state_dim, zs_dim, hdim, activ)
        # else:
        #     raise NotImplementedError("obs_space must be Box or Discrete")
            
        
        if isinstance(act_space, spaces.Box) and len(act_space.shape) == 1:
            self.act_dim = act_space.shape[0]
        elif isinstance(act_space, spaces.Discrete):
            self.act_dim = act_space.n
        else:
            raise NotImplementedError("act_space must be Box or Discrete")
    
            

        self.za = nn.Linear(self.act_dim, za_dim)
        self.zsa = BaseMLP(zs_dim + za_dim, zsa_dim, hdim, activ)
        self.model = nn.Linear(zsa_dim, num_bins + zs_dim + 1)

        self.zs_dim = zs_dim

        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, zs: torch.Tensor, action: torch.Tensor):
        za = self.activ(self.za(action))
        return self.zsa(torch.cat([zs, za], 1))


    def model_all(self, zs: torch.Tensor, action: torch.Tensor):
        zsa = self.forward(zs, action)
        dzr = self.model(zsa)
        return dzr[:,0:1], dzr[:,1:self.zs_dim+1], dzr[:,self.zs_dim+1:] # done, zs, reward


    def cnn_zs(self, state: torch.Tensor):
        state = state/255. - 0.5
        zs = self.activ(self.zs_cnn1(state))
        zs = self.activ(self.zs_cnn2(zs))
        zs = self.activ(self.zs_cnn3(zs))
        zs = self.activ(self.zs_cnn4(zs)).reshape(state.shape[0], -1)
        return ln_activ(self.zs_lin(zs), self.activ)


    def mlp_zs(self, state: torch.Tensor):
        return ln_activ(self.zs_mlp(state), self.activ)