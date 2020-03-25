#!/usr/bin/env python3
"""This is an example to train PPO on MT10 environment."""
# pylint: disable=no-value-for-parameter

import click
from metaworld.benchmarks import MT10
from metaworld.envs.mujoco.env_dict import EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.multi_env_wrapper import (
    MultiEnvWrapper,
    round_robin_strategy,
)
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO as TF_PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianMLPPolicy

MT10_envs_by_id = {
    task: env(*EASY_MODE_ARGS_KWARGS[task]['args'],
              **EASY_MODE_ARGS_KWARGS[task]['kwargs'])
    for (task, env) in EASY_MODE_CLS_DICT.items()
}


@click.command()
@click.option('--seed', default=1)
@click.option('--n_epochs', default=1000)
@click.option('--batch_size', default=256)
@click.option('--max_path_length', default=256)
@wrap_experiment(snapshot_mode='all')
def ppo_mt10(ctxt, seed, n_epochs, batch_size, max_path_length):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        n_epochs (int): Number of training epochs.
        batch_size (int): Number of batch size.
        max_path_length (int): Maximum path length.

    """
    set_seed(seed)

    with LocalTFRunner(ctxt) as runner:
        env = MultiEnvWrapper(normalize(MT10.get_train_tasks(),
                                        expected_action_scale=10.),
                              sample_strategy=round_robin_strategy)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=False,
                optimizer=FirstOrderOptimizer,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                    tf_optimizer_args=dict(learning_rate=3e-4),
                ),
            ),
        )

        algo = TF_PPO(env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      max_path_length=max_path_length,
                      discount=0.99,
                      gae_lambda=0.95,
                      center_adv=True,
                      lr_clip_range=0.2,
                      optimizer_args=dict(
                          batch_size=32,
                          max_epochs=10,
                          tf_optimizer_args=dict(learning_rate=3e-4)))

        runner.setup(algo, env)
        runner.train(n_epochs=n_epochs, batch_size=batch_size)


ppo_mt10()
