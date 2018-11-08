import numpy as np
import tensorflow as tf

from garage.envs import normalize
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.behavioral_cloning.point_env import PointEnv
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


def sample_from_experts(tasks, time_horizon=50, size_in_transitions=800):
    """Sample actions and observations from the expert

    We load the TensorFlow variables that define our expert from the TF session
    stored in the folder "ppo_point_expert", and sample actions, rewards, done
    states and observations from it, saving the results in a replay buffer and
    in numpy files in the folder point_data.

    Parameters:
        - tasks: dictionary of task IDs (must match the id in the TF session)
          and tuples of goals (coordinate in two dimensional space)
        - time_horizon: maximum number of transitions per episode
        - size_in_transitions: number of transitions to sample
    """

    for task_id, task in tasks.items():
        with tf.Graph().as_default():
            env = TfEnv(normalize(PointEnv(goal=task)))

            replay_buffer = SimpleReplayBuffer(
                env_spec=env.spec,
                size_in_transitions=size_in_transitions,
                time_horizon=time_horizon)

            policy = GaussianMLPPolicy(
                name="policy",
                env_spec=env.spec,
                hidden_sizes=(20, 20),
                hidden_nonlinearity=tf.nn.relu)

            with tf.Session() as sess:
                saver = tf.train.Saver()
                saver.restore(
                    sess, "./garage/tf/behavioral_cloning/" +
                    "ppo_point_expert/ppo_task_" + task_id + ".ckpt")
                while not replay_buffer.full:
                    done = False
                    obs = env.reset()
                    i = 0
                    while (not done and (not replay_buffer.full)
                           and not (i == time_horizon)):
                        env.render()
                        action, _ = policy.get_action(obs)
                        # To visualize how our clone model behaves, make a
                        # prediction from the clone and step based on the
                        # retrieved action.
                        next_obs, reward, done, info = env.step(action)
                        replay_buffer.add_transition(
                            action=[action],
                            observation=[obs],
                            terminal=[done],
                            reward=[reward],
                            next_observation=[next_obs])
                        obs = next_obs
                        i += 1
                env.close()
            data = replay_buffer.sample(size_in_transitions - 1)
            np.save(
                "garage/tf/behavioral_cloning/point_data/ppo_data_task_" +
                task_id + ".npy", data)
            # Retrieve data as:
            # expert = np.load("garage/tf/behavioral_cloning/expert_data.npy")
            # obs = expert[()]["observation"]
            # acts = expert[()]["action"]


def main():
    tasks = {"1": (0, 3), "2": (3, 0), "3": (0, -3), "4": (-3, 0)}
    sample_from_experts(tasks)


if __name__ == '__main__':
    main()
