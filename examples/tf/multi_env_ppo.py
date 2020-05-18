#!/usr/bin/env python3
"""This is an example to train multiple tasks with PPO algorithm."""
import gym
import numpy as np
import tensorflow as tf

from garage.envs import normalize, PointEnv
from garage.envs.multi_env_wrapper import MultiEnvWrapper
from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalMLPPolicy, ContinuousMLPPolicy, \
    GaussianMLPPolicy


def circle(r, n):
    """Generate n points on a circle of radius r.

    Args:
        r (float): Radius of the circle.
        n (int): Number of points to generate.

    Yields:
        tuple(float, float): Coordinate of a point.

    """
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
        yield r * np.sin(t), r * np.cos(t)


N = 1
goals = circle(3.0, N)
TASKS = {
    str(i + 1): {
        'args': [],
        'kwargs': {
            'goal': g,
            'never_done': False,
            'done_bonus': 0.0,
        }
    }
    for i, g in enumerate(goals)
}

def run_task(snapshot_config, *_):
    """Run task.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.

        _ (object): Ignored by this function.

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        tasks = TASKS
        task_names = sorted(tasks.keys())
        task_args = [tasks[t]['args'] for t in task_names]
        task_kwargs = [tasks[t]['kwargs'] for t in task_names]
        task_envs = [
            TfEnv(PointEnv(*t_args, **t_kwargs))
            for t_args, t_kwargs in zip(task_args, task_kwargs)
        ]
        env = MultiEnvWrapper(task_envs)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_nonlinearity=tf.nn.tanh,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=100,
                   discount=0.99,
                   gae_lambda=0.95,
                   lr_clip_range=0.2,
                   policy_ent_coeff=0.0,
                   optimizer_args=dict(
                       batch_size=32,
                       max_epochs=10,
                       tf_optimizer_args=dict(learning_rate=1e-3),
                   ))

        runner.setup(algo, env)
        runner.train(n_epochs=120, batch_size=2048, plot=False)


run_experiment(run_task, snapshot_mode='last', seed=1)
