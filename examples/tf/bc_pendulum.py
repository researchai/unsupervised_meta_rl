import gym
import tensorflow as tf
import numpy as np

from garage.misc.instrument import run_experiment
from garage.tf.algos import BC
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy


def run_task(*_):

    env = TfEnv(gym.make('InvertedDoublePendulum-v2'))
    policy = ContinuousMLPPolicy(
        env_spec=env.spec,
        name="Actor",  # Has to match actor policy name from DDPG-trained expert
        hidden_sizes=[100, 100],
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh)
    expert_dataset_file = "garage/tf/behavioral_cloning/expert_data2.npy"
    expert_tf_session_files = "garage/tf/behavioral_cloning/ddpg_expert/ddpg_model.ckpt"
    expert_dataset = np.load(expert_dataset_file)[()]

    bc = BC(
        env=env,
        policy=policy,
        expert_dataset=expert_dataset,
        max_path_length=1024,
        rollout_batch_size=10,
        policy_lr=4e-3,
        n_epochs=50,
        #        GET_EXPERT_REWARDS=True,
        expert_tf_session=expert_tf_session_files)

    bc.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
