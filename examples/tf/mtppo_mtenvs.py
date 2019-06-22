# flake8: noqa
#!/usr/bin/env python3
"""
This is an example to train a multi env task with PPO algorithm.
"""

import gym
import tensorflow as tf

from garage.envs import normalize
from garage.envs.env_spec import EnvSpec
from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.samplers import MultiEnvironmentVectorizedSampler


def make_envs(env_names):
    return [TfEnv(normalize(gym.make(env_name))) for env_name in env_names]


def run_task(*_):
    with LocalRunner() as runner:
        envs = make_envs(['InvertedDoublePendulum-v2'] * 3)

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

        runner.setup(algo, envs, sampler_cls=MultiEnvironmentVectorizedSampler)

        runner.train(n_epochs=120, batch_size=2048 * len(envs), plot=False)


run_experiment(run_task, snapshot_mode='last', seed=1)
