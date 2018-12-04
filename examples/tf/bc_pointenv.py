from types import SimpleNamespace

import tensorflow as tf
import numpy as np

from garage.misc.instrument import run_experiment
from garage.tf.algos import BC
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.policies import GaussianMLPPolicy
from garage.envs import PointEnv


def run_task(v):
    v = SimpleNamespace(**v)

    env = TfEnv(PointEnv(goal=v.goal))
    expert_dataset = np.load(v.expert_dataset_path)[()]

    if v.stochastic_policy:
        policy = GaussianMLPPolicy(
            env_spec=env.spec, name="Target", hidden_sizes=[20, 20])
    else:
        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            name="Target",
            hidden_sizes=[20, 20],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

    bc = BC(
        env=env,
        policy=policy,
        expert_dataset=expert_dataset,
        stochastic_policy=v.stochastic_policy,
        max_path_length=100,
        rollout_batch_size=10,
        policy_lr=1e-4,
        n_epochs=10)

    bc.train()


config = dict(
    stochastic_policy=False,
    expert_dataset_path="data/bc/point_data/ppo_data_task_1.npy",
    goal=(0., 3.),
)

run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    variant=config,
    plot=False,
)
