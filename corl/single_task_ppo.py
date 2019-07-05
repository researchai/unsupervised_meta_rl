#!/usr/bin/env python3
"""
This is an example to train a task with PPO algorithm.
Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.
Results:
    AverageDiscountedReturn: 500
    RiseTime: itr 40
"""
import argparse

import gym
import numpy as np
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import LocalRunner, run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline

from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy

from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from hard_env_list import TRAIN_DICT, TRAIN_ARGS_KWARGS

EXP_PREFIX = 'single_task_{}'
SAWYER_ENV = None

def run_task(*_):
    with LocalRunner() as runner:
        EXP_PREFIX = EXP_PREFIX.format(type(SAWYER_ENV).__name__)
        print(EXP_PREFIX)

        env = TfEnv(SAWYER_ENV)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(200, 100),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            adaptive_std=True,
            init_std=2.,
        )

        # baseline = GaussianMLPBaseline(
        #     env_spec=env.spec,
        #     regressor_args=dict(
        #         hidden_sizes=(200, 100),
        #         use_trust_region=True,
        #     ),
        # )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        # NOTE: make sure when setting entropy_method to 'max', set
        # center_adv to False and turn off policy gradient. See
        # tf.algos.NPO for detailed documentation.
        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=150,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
            ),
            stop_entropy_gradient=False,
            entropy_method='regularized',
            policy_ent_coeff=1e-3,
            center_adv=True,
            # use_softplus_entropy=True,
        )

        runner.setup(algo, env)

        runner.train(n_epochs=10000, batch_size=4096, plot=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single task PPO.')
    parser.add_argument(
        '--idx',
        type=int,
        default=None,
        help='Index of environment.')
    args = parser.parse_args()
    env_idx = args.idx

    env_cls = TRAIN_DICT[env_idx]
    env_args = TRAIN_ARGS_KWARGS[env_idx]['args']
    env_kwargs = TRAIN_ARGS_KWARGS[env_idx]['kwargs']
    print(env_args, env_kwargs)
    SAWYER_ENV = env_cls(*env_args, **env_kwargs)

    run_experiment(run_task, exp_prefix=EXP_PREFIX, snapshot_mode='all', seed=1)
