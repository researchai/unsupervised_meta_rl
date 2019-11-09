#!/usr/bin/env python3
"""
This is an example to train a task with TRPO algorithm.

It uses an LSTM-based recurrent policy. To use a GRU-based recurrent
policy, swap the commented lines. Here it runs CartPole-v1 environment
with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 13

"""
from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO2
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import CategoricalGRUPolicy2


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy = CategoricalGRUPolicy2(
            name='policy',
            env_spec=env.spec
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO2(env_spec=env.spec,
                     policy=policy,
                     baseline=baseline,
                     max_path_length=100,
                     discount=0.99,
                     max_kl_step=0.01,
                     optimizer=ConjugateGradientOptimizer,
                     optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                        base_eps=1e-5)))

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
