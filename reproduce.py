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
"""Example of training a policy from exp-x config with OmniSafe."""

import warnings

import torch

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Benchmark_Safety_Velocity')

    # Set the algorithms.
    base_policy = ['PPO', 'MRQ', 'PPOLag', 'FOCOPS', 'PolicyGradient', 'P3O']

    # Set the environments.
    mujoco_envs = [
        'SafetyAntVelocity-v1',
        'SafetyHopperVelocity-v1',
        'SafetyHumanoidVelocity-v1',
        'SafetyWalker2dVelocity-v1',
        'SafetyHalfCheetahVelocity-v1',
    ]
    eg.add('env_id', mujoco_envs)

    # Set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0,1]
    # if you want to use CPU, please set gpu_id = None
    # gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None
    eg.add('logger_cfgs:wandb_project', ["DS699_400_all"])
    eg.add('algo', base_policy)
    eg.add('logger_cfgs:use_wandb', [True])
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [32])
    eg.add('algo_cfgs:steps_per_epoch', [2000])
    eg.add('train_cfgs:total_steps', [2000 * 400])
    eg.add('seed', [0, 5])
    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=120, gpu_id=gpu_id)

    # just fill in the name of the parameter of which value you want to compare.
    # then you can specify the value of the parameter you want to compare,
    # or you can just specify how many values you want to compare in single graph at most,
    # and the function will automatically generate all possible combinations of the graph.
    # but the two mode can not be used at the same time.
    # eg.analyze(parameter='env_id', values=None, compare_num=1, cost_limit=25)
    # eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    eg.evaluate(num_episodes=1)
