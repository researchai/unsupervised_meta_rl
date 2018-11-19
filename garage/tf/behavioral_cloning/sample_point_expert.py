import sys

import numpy as np
import tensorflow as tf

from garage.envs import normalize
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.behavioral_cloning.point_env import PointEnv
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


def sample_from_experts_random(tasks, time_horizon=50,
                               size_in_transitions=800):
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


def sample_from_experts_sequential(tasks, num_trajectories=50,
                                   time_horizon=50):
    """Sample actions and observations from the expert

    We load the TensorFlow variables that define our expert from the TF session
    stored in the folder "ppo_point_expert", and sample actions, rewards, done
    states and observations from it, saving the results in a numpy structured
    array and in the numpy file at
    "garage/tf/behavioral_cloning/point_data_sequential.npy".
    The data is divided by a task ID, and each task is divided by trajectories.
    Each trajectory contains a sequence of tuples with values (action,
    observation, terminal, reward, next_observation).
    Follow the next example to retrieve the data from the saved filed:
    ```
    import numpy as np

    data = np.load("garage/tf/behavioral_cloning/point_data_sequential.npy")
    task_1 = data.item().get("task_1")
    trajectory = task_1[0]  # Get the first trajectory from task 1
    print(trajectory["observation"])
    ```


    Parameters:
        - tasks: dictionary of task IDs (must match the id in the TF session)
          and tuples of goals (coordinate in two dimensional space)
        - num_trajectories: number of all trajectories sampled. This number is
          divided as evenly as possible amongst all tasks. For example, if the
          number of trajectories is 50 and there are four tasks, then 13
          trajectories will sampled for 2 tasks, and 12 trajectories for the
          remaining 2.
        - time_horizon: maximum number of transitions per trajectory
    """

    num_tasks = len(tasks.items())
    remaining_trajectories = num_trajectories % num_tasks

    all_tasks = {}
    for task_id, task in tasks.items():
        with tf.Graph().as_default():
            env = TfEnv(normalize(PointEnv(goal=task)))
            dt = np.dtype(
                [("action", env.spec.action_space.low.dtype,
                  env.spec.action_space.shape),
                 ("observation", env.spec.observation_space.low.dtype,
                  env.spec.observation_space.shape), ("terminal", np.bool),
                 ("reward", np.float32), ("next_observation",
                                          env.spec.observation_space.low.dtype,
                                          env.spec.observation_space.shape)])

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

                num_trajectories_per_task = int(num_trajectories / num_tasks)
                if remaining_trajectories:
                    num_trajectories_per_task += 1
                    remaining_trajectories -= 1

                completed_trajec = 0
                trajec_per_task = np.zeros(
                    (num_trajectories_per_task, time_horizon), dtype=dt)

                while completed_trajec != num_trajectories_per_task:
                    done = False
                    obs = env.reset()
                    i = 0
                    while (not done and not (i == time_horizon)):
                        env.render()
                        action, _ = policy.get_action(obs)
                        next_obs, reward, done, info = env.step(action)
                        trajec_per_task[completed_trajec][i] = (action, obs,
                                                                done, reward,
                                                                next_obs)
                        obs = next_obs
                        i += 1
                    completed_trajec += 1
            all_tasks.update({"task_" + task_id: trajec_per_task})
            env.close()
    np.save("garage/tf/behavioral_cloning/point_data_sequential.npy",
            all_tasks)


def main(mode):
    tasks = {"1": (0, 3), "2": (3, 0), "3": (0, -3), "4": (-3, 0)}
    if (mode == "-r"):
        sample_from_experts_random(tasks)
    elif (mode == "-s"):
        sample_from_experts_sequential(tasks)
    else:
        print("The option %s is invalid." % mode)
        print_help()


def print_help():
    print("Please specify one of these options:")
    print("    -r: sample randomly")
    print("    -s: sample sequentially")


if __name__ == '__main__':
    if (len(sys.argv) > 1):
        main(sys.argv[1])
    else:
        print_help()
