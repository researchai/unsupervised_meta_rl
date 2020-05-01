#!/usr/bin/env python3
"""DDPG example with pixel observations using InvertedDoublePendulum-v2."""

import click
import gym
import numpy as np
import tensorflow as tf

from garage import wrap_experiment
from garage.envs.wrappers import Grayscale
from garage.envs.wrappers import MaxAndSkip
from garage.envs.wrappers import PixelObservation
from garage.envs.wrappers import Resize
from garage.envs.wrappers import StackFrames
from garage.experiment import run_experiment
from garage.experiment.deterministic import set_seed
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousCNNPolicy
from garage.tf.q_functions import ContinuousCNNQFunction



@click.command()
@click.option('--buffer_size', type=int, default=int(5e4))
@click.option('--seed', type=int, default=1)
@wrap_experiment(snapshot_mode='last')
def ddpg_pendulum_pixels(ctxt=None, seed=1, buffer_size=int(1e4)):
  
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt, max_cpus=4) as runner:
        
        env = gym.make('Pendulum-v0')
        env.reset() # fixes bug when using gym.wrappers.PixelObservation and Pendulum
        env = PixelObservation(env)
        env = Grayscale(env)
        env = Resize(env, 86, 86)
        env = StackFrames(env, 2)
        
        env = TfEnv(env, is_image=True)

        action_noise = OUStrategy(env.spec, sigma=0.2)

        policy = ContinuousCNNPolicy(env_spec=env.spec,
                                     filter_dims=(8, 4, 3),
                                     num_filters=(32, 64, 64),
                                     strides=(4, 2, 1),
                                     hidden_sizes=(64, 64),
                                     hidden_nonlinearity=tf.nn.relu,
                                     output_nonlinearity=tf.nn.tanh)
        qf = ContinuousCNNQFunction(env_spec=env.spec,
                                    filter_dims=(8, 4, 3),
                                    num_filters=(32, 64, 64),
                                    strides=(4, 2, 1),
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=tf.nn.relu,
                                    action_merge_layer=0)

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=buffer_size,
            time_horizon=1)

        ddpg = DDPG(env_spec=env.spec,
                    policy=policy,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    max_path_length=100,
                    steps_per_epoch=20,
                    target_update_tau=1e-2,
                    n_train_steps=50,
                    discount=0.99,
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer,
                    flatten_obses=False)

        runner.setup(algo=ddpg, env=env)

        runner.train(n_epochs=500, batch_size=100)

ddpg_pendulum_pixels()