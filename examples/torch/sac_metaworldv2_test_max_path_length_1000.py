#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import click
import gym
from metaworld.envs.mujoco.env_dict import ALL_ENVIRONMENTS
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
import garage.torch.utils as tu

_SEED = None
_GPU = None
_ENV_NAME = ""

@click.command()
@click.option('--seed', '_seed', type=int, default=np.random.randint(0, 1000))
@click.option('--gpu', '_gpu', type=int, default=None)
@click.option('--env', '_env_name', type=str, default='assembly-v1')
def get_args(_seed=1, _gpu=None, _env_name=None):
    """Retrieve args from command line.

    Args:
        _seed (int): Used to seed the random number generator to produce
            determinism.
        _gpu (int): The ID of the gpu to be used (used on multi-gpu machines).
        _env_name (str): The name of the environment being tested

    """
    global _SEED, _GPU, _ENV_NAME
    _SEED = _seed
    _GPU = _gpu
    _ENV_NAME = _env_name
    task_types = ['pick-place', 'reach', 'push']
    @wrap_experiment(snapshot_mode='gap', prefix=_ENV_NAME, snapshot_gap=25)
    def sac_metaworldv2_test_max_path_length_1000(ctxt=None):
        """Set up environment and algorithm and run the task.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by LocalRunner to create the snapshotter.

        """
        global _SEED, _GPU, _ENV_NAME
        not_in_mw = "the env_name specified is not a metaworld environment"
        assert _ENV_NAME in ALL_ENVIRONMENTS, not_in_mw
        deterministic.set_seed(_SEED)
        runner = LocalRunner(snapshot_config=ctxt)
        env_cls = ALL_ENVIRONMENTS[_ENV_NAME]
        task_type = None
        for name in task_types:
            if name in _ENV_NAME:
                task_type = name
            if task_type is 'pick-place':
                task_type = 'pick_place'

        if task_type:
            env = GarageEnv(normalize(env_cls(random_init=False, task_type=task_type)))
        else:
            env = GarageEnv(normalize(env_cls(random_init=False)))
        env.env.env.max_path_length = 1000
        policy = TanhGaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=[256, 256],
            hidden_nonlinearity=nn.ReLU,
            output_nonlinearity=None,
            min_std=np.exp(-20.),
            max_std=np.exp(2.),
        )

        qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[256, 256],
                                    hidden_nonlinearity=F.relu)

        qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[256, 256],
                                    hidden_nonlinearity=F.relu)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
        batch_size = 1000

        sac = SAC(env_spec=env.spec,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                gradient_steps_per_itr=batch_size,
                max_path_length=1000,
                replay_buffer=replay_buffer,
                min_buffer_size=1e4,
                target_update_tau=5e-3,
                discount=0.99,
                buffer_batch_size=512,
                reward_scale=1.,
                steps_per_epoch=20,
                num_evaluation_trajectories=10)

        if _GPU is not None:
            tu.set_gpu_mode(True, _GPU)
        sac.to()
        runner.setup(algo=sac, env=env, sampler_cls=LocalSampler)
        runner.train(n_epochs=150, batch_size=batch_size)
    sac_metaworldv2_test_max_path_length_1000()

get_args()
