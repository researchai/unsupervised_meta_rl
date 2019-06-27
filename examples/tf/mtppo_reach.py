# flake8: noqa
#!/usr/bin/env python3
"""
This is an example to train a multi env task with PPO algorithm.
"""

import gym
import tensorflow as tf
import numpy as np

from garage.envs import normalize
from garage.envs.env_spec import EnvSpec
from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.samplers import MultiEnvironmentVectorizedSampler2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv

EXP_PREFIX = 'ppo_reach_multi_task'
def make_envs(env_names):
    return [TfEnv(normalize(gym.make(env_name))) for env_name in env_names]


def run_task(*_):
    with LocalRunner() as runner:

        goal_low = np.array((-0.1, 0.8, 0.05))
        goal_high = np.array((0.1, 0.9, 0.3))
        goals = np.random.uniform(low=goal_low, high=goal_high, size=(4, len(goal_low))).tolist()
        print('constructing envs')
        envs = [
            TfEnv(SawyerReachPushPickPlace6DOFEnv(
                tasks=[{'goal': np.array(g),  'obj_init_pos': np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'reach'}],
                random_init=False,
                if_render=False,))
            for g in goals
        ]

        policy = GaussianMLPPolicy(
            env_spec=envs[0].spec,
            task_dim=len(envs),
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=envs[0].spec,
            task_dim=len(envs),
            regressor_args=dict(
                hidden_sizes=(32, 32),
                use_trust_region=True,
            ),
        )

        # NOTE: make sure when setting entropy_method to 'max', set
        # center_adv to False and turn off policy gradient. See
        # tf.algos.NPO for detailed documentation.
        algo = PPO(
            env_spec=envs[0].spec,
            task_dim=len(envs),
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
            ),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=0.02,
            center_adv=False,
        )

        runner.setup(algo, envs, sampler_cls=MultiEnvironmentVectorizedSampler2)

        runner.train(n_epochs=500, batch_size=2048 * len(envs), plot=False)


run_experiment(run_task, exp_prefix=EXP_PREFIX, seed=1)
