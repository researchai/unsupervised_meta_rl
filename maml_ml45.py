#!/usr/bin/env python3
"""This script creates a regression test over garage-MAML and ProMP-TRPO.

Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length. We also need to change the
garage.tf.samplers.BatchSampler to smooth the reward curve.
"""
import click
import metaworld.benchmarks
import torch

from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import deterministic, LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy

# Same as promp:full_code/config/trpo_maml_config.json
hyper_parameters = {
    'hidden_sizes': [100, 100],
    'max_kl': 0.01,
    'inner_lr': 0.1,
    'gae_lambda': 1.0,
    'discount': 0.99,
    'max_path_length': 150,
    'fast_batch_size': 10,  # num of rollouts per task
    'meta_batch_size': 20,  # num of tasks
    'n_epochs': 1700,
    # 'n_epochs': 1,
    'n_trials': 3,
    'num_grad_update': 1,
    'n_parallel': 1,
    'inner_loss': 'log_likelihood'
}


@click.command()
@click.option('--seed', default=1, help='Seed to control determinism.')
@wrap_experiment
def run_maml(ctxt=None, seed=1):
    """Create garage PyTorch MAML model and training.

    Args:
        ctxt (garage.experiment.ExperimentContxt): The experiment contxt.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)
    meta_env = metaworld.benchmarks.ML45.get_train_tasks()
    env = GarageEnv(normalize(meta_env, expected_action_scale=10.))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hyper_parameters['hidden_sizes'],
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = MAMLTRPO(env=env,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=hyper_parameters['max_path_length'],
                    discount=hyper_parameters['discount'],
                    gae_lambda=hyper_parameters['gae_lambda'],
                    meta_batch_size=hyper_parameters['meta_batch_size'],
                    inner_lr=hyper_parameters['inner_lr'],
                    max_kl_step=hyper_parameters['max_kl'],
                    num_grad_updates=hyper_parameters['num_grad_update'])

    runner = LocalRunner(snapshot_config=ctxt)
    runner.setup(algo, env, sampler_args=dict(n_envs=5))
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=(hyper_parameters['fast_batch_size'] *
                             hyper_parameters['max_path_length']))


if __name__ == '__main__':
    run_maml()
