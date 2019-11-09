#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
import gym

from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import GaussianLSTMPolicy


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('LunarLanderContinuous-v2'))

        policy = GaussianLSTMPolicy(env_spec=env.spec)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                    discount=0.99,
                    max_kl_step=0.01,
                    optimizer=ConjugateGradientOptimizer,
                    optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                        base_eps=1e-5)))

        runner.setup(algo, env)
        runner.train(n_epochs=40, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
