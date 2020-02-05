#!/usr/bin/env python3
"""
This is an example to train a task with DDPG algorithm written in PyTorch.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

"""
import numpy as np

import gym
import torch
from torch.nn import functional as F  # NOQA
from torch import nn as nn

from garage import wrap_experiment
from garage.envs import normalize, normalize_reward
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT
from metaworld.benchmarks import MT50
from garage.envs import GarageEnv
from garage.envs.multi_task_metaworld_wrapper import MTMetaWorldWrapper

from garage.experiment import LocalRunner, run_experiment
from garage.replay_buffer import SimpleReplayBuffer, SACReplayBuffer
from garage.torch.algos import MTSAC
from garage.torch.policies import TanhGaussianMLPPolicy2
from garage.torch.q_functions import ContinuousMLPQFunction

from garage.sampler import SimpleSampler
import garage.torch.utils as tu
import argparse

@wrap_experiment(snapshot_mode='none')
def mt50_normalize_rewards_RI_sac(ctxt=None, seed=1):
    """Set up environment and algorithm and run the task."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int,default=0,help='gpu id')
    options = parser.parse_args()
    runner = LocalRunner(ctxt)
    for key in ['train', 'test']:
        for env_name, args in HARD_MODE_ARGS_KWARGS[key].items():
            HARD_MODE_ARGS_KWARGS[key][env_name]['kwargs']['random_init'] = False 
    envs = MT50.get_train_tasks(sample_all=True)
    test_envs = MT50.get_test_tasks(sample_all=True)
    MT50_envs_by_id = {name: GarageEnv(normalize_reward(env)) for (name,env) in zip (envs._task_names, envs._task_envs)}
    MT50_envs_test = {name: GarageEnv(env) for (name,env) in zip (test_envs._task_names, test_envs._task_envs)}
    env = MTMetaWorldWrapper(MT50_envs_by_id)

    policy = TanhGaussianMLPPolicy2(env_spec=env.spec,
                               hidden_sizes=[400, 400, 400],
                               hidden_nonlinearity=nn.ReLU,
                               output_nonlinearity=None,
                               min_std=np.exp(-20.),
                               max_std=np.exp(2.),)

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[400, 400, 400],
                                hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[400, 400, 400],
                                hidden_nonlinearity=F.relu)

    replay_buffer = SACReplayBuffer(env_spec=env.spec,
                                       max_size=int(1e6))
    sampler_args = {'agent': policy, 'max_path_length': 150}

    timesteps = 100000000
    batch_size = int(150 * env.num_tasks)
    num_evaluation_points = 500
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    sac = MTSAC(env=env,
                eval_env_dict=MT50_envs_test,
                env_spec=env.spec,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                gradient_steps_per_itr=250,
                epoch_cycles=epoch_cycles,
                use_automatic_entropy_tuning=True,
                replay_buffer=replay_buffer,
                min_buffer_size=7500,
                target_update_tau=5e-3,
                discount=0.99,
                buffer_batch_size=6400)
    tu.set_gpu_mode(True, options.gpu)
    sac.to()

    runner.setup(algo=sac, env=env, sampler_cls=SimpleSampler, sampler_args=sampler_args)

    runner.train(n_epochs=epochs, batch_size=batch_size)

s = np.random.randint(0, 1000)
mt50_normalize_rewards_RI_sac(seed=s)
