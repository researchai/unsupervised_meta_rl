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

from garage.envs import normalize
from metaworld.envs.mujoco.env_dict import EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT
from garage.envs import GarageEnv
from garage.envs.multi_task_metaworld_wrapper import MTMetaWorldWrapper

from garage.experiment import LocalRunner, run_experiment
from garage.replay_buffer import SimpleReplayBuffer, SACReplayBuffer
from garage.torch.algos import MTSAC
from garage.torch.policies import TanhGaussianMLPPolicy2
from garage.torch.q_functions import ContinuousMLPQFunction

from garage.sampler import SimpleSampler
import garage.torch.utils as tu


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task."""
    runner = LocalRunner(snapshot_config)
    MT10_envs_by_id = {}
    MT10_envs_test = {}
    for (task, env) in EASY_MODE_CLS_DICT.items():
        MT10_envs_by_id[task] = GarageEnv(env(*EASY_MODE_ARGS_KWARGS[task]['args'],
                                    **EASY_MODE_ARGS_KWARGS[task]['kwargs']))
        # python 3.6 dicts are ordered
        MT10_envs_test[task] = GarageEnv(env(*EASY_MODE_ARGS_KWARGS[task]['args'],
                                    **EASY_MODE_ARGS_KWARGS[task]['kwargs']))

    env = MTMetaWorldWrapper(MT10_envs_by_id)

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
    sac = MTSAC(env=env,
                eval_env_dict=MT10_envs_test,
                env_spec=env.spec,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                gradient_steps_per_itr=150,
                use_automatic_entropy_tuning=True,
                replay_buffer=replay_buffer,
                min_buffer_size=1500,
                target_update_tau=5e-3,
                discount=0.99,
                buffer_batch_size=1280)
    tu.set_gpu_mode(True)
    sac.to()
    runner.setup(algo=sac, env=env, sampler_cls=SimpleSampler, sampler_args=sampler_args)

    runner.train(n_epochs=13000, batch_size=1500)

run_experiment(
    run_task,
    snapshot_mode='last',
    seed=532,
)
