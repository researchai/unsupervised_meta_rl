import gym
import tensorflow as tf
import numpy as np

from garage.misc.instrument import run_experiment
from garage.tf.algos import BC, BCExpertEvaluator
from garage.tf.algos.bc import sample_from_expert
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy


def run_bc(*_):
    env = TfEnv(gym.make('InvertedDoublePendulum-v2'))
    expert_dataset_path = "data/bc/inverted_double_pendulum_expert_data.npz"
    # Load expert dataset, or generate it from an expert policy if not found
    try:
        expert_dataset = np.load(expert_dataset_path)
    except FileNotFoundError:
        # Policy has to exactly match actor policy graph from trained expert
        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            name="Actor",
            hidden_sizes=[64, 64],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)
        expert_tf_session_ckpt = "data/bc/ddpg_expert/ddpg_model.ckpt"
        expert_dataset = sample_from_expert(
            env=env,
            expert_policy=policy,
            expert_tf_session=expert_tf_session_ckpt,
            dataset_save_path=expert_dataset_path)

    tf.reset_default_graph()
    policy = ContinuousMLPPolicy(
        env_spec=env.spec,
        name="Target",
        hidden_sizes=[100, 100],
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh)

    bc = BC(
        env=env,
        policy=policy,
        expert_dataset=expert_dataset,
        max_path_length=1024,
        rollout_batch_size=10,
        policy_lr=4e-3,
        n_epochs=50)

    bc.train()


def run_eval(*_):
    tf.reset_default_graph()
    env = TfEnv(gym.make('InvertedDoublePendulum-v2'))
    # Policy has to exactly match actor policy graph from trained expert
    policy = ContinuousMLPPolicy(
        env_spec=env.spec,
        name="Actor",
        hidden_sizes=[64, 64],
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh)
    expert_tf_session_ckpt = "data/bc/ddpg_expert/ddpg_model.ckpt"

    exp_eval = BCExpertEvaluator(
        env=env, policy=policy, expert_tf_session=expert_tf_session_ckpt)

    exp_eval.train()


run_experiment(
    run_bc,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)

run_experiment(
    run_eval,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
