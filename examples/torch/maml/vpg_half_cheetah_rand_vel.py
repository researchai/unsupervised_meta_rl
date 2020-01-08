"""This script creates a regression test over garage-MAML and ProMP-TRPO.

Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length. We also need to change the
garage.tf.samplers.BatchSampler to smooth the reward curve.
"""
import datetime
import os.path as osp
import random

import numpy as np
import dowel
from dowel import logger as dowel_logger
import pytest
import torch
import tensorflow as tf

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import deterministic, LocalRunner, SnapshotConfig, run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.algos import VPG, MAML
from garage.envs import HalfCheetahVelEnv
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.optimizers import ConjugateGradientOptimizer

from tests import benchmark_helper
import tests.helpers as Rh

hyper_parameters = {
    'hidden_sizes': [100, 100],
    'max_kl': 0.01,
    'inner_lr': 0.1,
    'gae_lambda': 1.0,
    'discount': 0.99,
    'max_path_length': 100,
    'fast_batch_size': 20,
    'meta_batch_size': 20,  # num of tasks
    'n_epochs': 500,
    'n_trials': 2,
    'num_grad_update': 1,
    'n_parallel': 1,
    'inner_loss': 'log_likelihood'
}


def run_garage(snapshot_config, *_):
    env = GarageEnv(normalize(HalfCheetahVelEnv()))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hyper_parameters['hidden_sizes'],
        hidden_nonlinearity=torch.relu,
        output_nonlinearity=None,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    meta_optimizer = (ConjugateGradientOptimizer, {
        'max_constraint_value': hyper_parameters['max_kl']
    })

    inner_algo = VPG(env_spec=env.spec,
                     policy=policy,
                     baseline=baseline,
                     max_path_length=hyper_parameters['max_path_length'],
                     discount=hyper_parameters['discount'],
                     gae_lambda=hyper_parameters['gae_lambda'])

    algo = MAML(env=env,
                policy=policy,
                baseline=baseline,
                meta_batch_size=hyper_parameters['meta_batch_size'],
                inner_lr=hyper_parameters['inner_lr'],
                inner_algo=inner_algo,
                num_grad_updates=hyper_parameters['num_grad_update'],
                meta_optimizer=meta_optimizer)

    runner = LocalRunner(snapshot_config=snapshot_config)
    runner.setup(algo=algo, env=env)
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=(hyper_parameters['fast_batch_size'] *
                             hyper_parameters['max_path_length']))


run_experiment(run_garage, snapshot_mode='last', seed=1)
