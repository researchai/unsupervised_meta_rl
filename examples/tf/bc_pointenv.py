import tensorflow as tf
import numpy as np

from garage.misc.instrument import run_experiment
from garage.tf.algos import BC
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.behavioral_cloning.point_env import PointEnv


def run_task(*_):
    env = TfEnv(PointEnv(goal=(0, 3)))
    expert_dataset_path = f"data/bc/point_data/ppo_data_task_2.npy"
    expert_dataset = np.load(expert_dataset_path)[()]

    tf.reset_default_graph()
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
        max_path_length=100,
        rollout_batch_size=10,
        # policy_lr=1e-4,
        n_epochs=10)

    bc.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
