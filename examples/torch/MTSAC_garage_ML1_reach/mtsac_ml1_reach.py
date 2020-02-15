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
from garage.envs import GarageEnv
from garage.envs.multi_task_metaworld_wrapper import MTMetaWorldWrapper
from garage.envs import normalize_reward
from garage.envs import ML1WithPinnedGoal
from garage.experiment import LocalRunner, run_experiment
from garage.replay_buffer import SimpleReplayBuffer, SACReplayBuffer
from garage.torch.algos import MTSAC
from garage.torch.policies import TanhGaussianMLPPolicy2
from garage.torch.q_functions import ContinuousMLPQFunction
import argparse
from garage.sampler import SimpleSampler
import garage.torch.utils as tu

def get_ML1_envs(name):
    bench = ML1WithPinnedGoal.get_train_tasks(name)
    tasks = [{'task': 0, 'goal': i} for i in range(50)]
    ret = {}
    for task in tasks:
        new_bench = bench.clone(bench)
        new_bench.set_task(task)
        ret[("goal"+str(task['goal']))] = GarageEnv(normalize_reward(new_bench.active_env))
    return ret

def get_ML1_envs_test(name):
    bench = ML1WithPinnedGoal.get_train_tasks(name)
    tasks = [{'task': 0, 'goal': i} for i in range(50)]
    ret = {}
    for task in tasks:
        new_bench = bench.clone(bench)
        new_bench.set_task(task)
        ret[("goal"+str(task['goal']))] = GarageEnv((new_bench.active_env))
    return ret

@wrap_experiment(snapshot_mode='none')
def mtsac_ml1_reach(ctxt=None, seed=1):
    """Set up environment and algorithm and run the task."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    options = parser.parse_args()
    runner = LocalRunner(ctxt)
    Ml1_reach_envs = get_ML1_envs("pick-place-v1")
    Ml1_reach_test_envs = get_ML1_envs_test("pick-place-v1")
    env = MTMetaWorldWrapper(Ml1_reach_envs)

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

    timesteps = 20000000
    batch_size = int(150 * env.num_tasks)
    num_evaluation_points = 500
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    sac = MTSAC(env=env,
                eval_env_dict=Ml1_reach_test_envs,
                env_spec=env.spec,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                gradient_steps_per_itr=150,
                epoch_cycles=epoch_cycles,
                use_automatic_entropy_tuning=True,
                replay_buffer=replay_buffer,
                min_buffer_size=1500,
                target_update_tau=5e-3,
                discount=0.99,
                buffer_batch_size=1280)
    tu.set_gpu_mode(True, options.gpu)
    sac.to()

    runner.setup(algo=sac, env=env, sampler_cls=SimpleSampler, sampler_args=sampler_args)

    runner.train(n_epochs=epochs, batch_size=batch_size)

s = np.random.randint(0, 1000)
mtsac_ml1_pick_place(seed=s)
