import gym

import numpy as np
import tensorflow as tf

from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy


def sample_from_expert(env, size_in_transitions=800, time_horizon=200):
    """Sample actions and observations from the expert

    We load the TensorFlow variables that define our expert from the TF session
    stored in the folder "ddpg_expert", and sample actions and observations
    from it, saving the results in a replay buffer.

    Parameters:
        - env: the gym environment
        - buffer_size: number of samples that our replay buffer will contain
    Returns:
        - replay_buffer: a dictionary with arrays of observations and actions
          with size defined by buffer_size.
    """

    buffer_shapes = {
        "action": env.action_space.shape,
        "observation": env.observation_space.shape
    }
    replay_buffer = SimpleReplayBuffer(
        env_spec=env.spec,
        size_in_transitions=size_in_transitions,
        time_horizon=time_horizon)

    with tf.Graph().as_default():
        actor_net = ContinuousMLPPolicy(
            env_spec=env,
            name="Actor",
            hidden_sizes=[64, 64],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            expert_tf_session = ("garage/tf/behavioral_cloning/ddpg_expert/"
                                 "ddpg_model.ckpt")
            saver.restore(sess, expert_tf_session)
            while not replay_buffer.full:
                done = False
                obs = env.reset()
                samples = 0
                while not done and (not replay_buffer.full):
                    action, _ = actor_net.get_action(obs)
                    # To visualize how our clone model behaves, make a
                    # prediction from the clone and step based on the retrieved
                    # action.
                    next_obs, reward, done, info = env.step(action)
                    samples += 1
                    replay_buffer.add_transition(
                        action=[action],
                        observation=[obs],
                        terminal=[done],
                        reward=[reward],
                        next_observation=[next_obs])
                    obs = next_obs
                print("Samples collected by episode in the expert", samples)
            env.close()

    data = replay_buffer.sample(size_in_transitions - 1)
    np.save("garage/tf/behavioral_cloning/expert_data.npy", data)
    # Retrieve data as:
    # expert = np.load("garage/tf/behavioral_cloning/expert_data.npy")
    # obs = expert[()]["observation"]
    # acts = expert[()]["action"]

    return replay_buffer


def optimize_nn(predicted_action_ph):
    """Build the optimizer for the clone of our expert.

    It reduces the loss between the output for the actions of the clone and the
    the actions provided by the expert in the buffer.

    Parameters:
        - predicted_action_ph: placeholder for the predicted actions.
    Returns:
        - minimize_sym: obtain the symbolic operation to minimize the loss in
          our clone with respect the samples from the expert.
        - loss_sym: obtain the symbolic operation for the loss.
        - expert_action_ph: the placeholder for the expert actions. This is one
          of the two inputs to calculate the loss in our optimizer.
    """
    expert_action_ph = tf.placeholder(
        tf.float32, shape=predicted_action_ph.shape)

    with tf.variable_scope("loss"):
        loss_sym = tf.squared_difference(expert_action_ph, predicted_action_ph)
        loss_sym = tf.reduce_mean(loss_sym)

    with tf.variable_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        minimize_sym = optimizer.minimize(loss=loss_sym)

    return minimize_sym, loss_sym, expert_action_ph


def main():
    env_id = "InvertedDoublePendulum-v2"
    env = TfEnv(gym.make(env_id))
    replay_buffer = sample_from_expert(env)
    clone_net = ContinuousMLPPolicy(
        env_spec=env,
        name="Clone",
        hidden_sizes=[64, 64],
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh)

    obs_shape = [None] + list(env.spec.observation_space.shape)
    obs_ph = tf.placeholder(tf.float32, shape=obs_shape)
    action_ph = clone_net.get_action_sym(obs_ph)
    minimize_sym, loss_sym, expert_action_ph = optimize_nn(action_ph)

    steps_print_loss = 100
    num_steps = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            done = False
            obs = env.reset()
            while not done:
                env.render()

                samples = replay_buffer.sample(128)
                clone_actions = clone_net.get_actions(samples["observation"])
                _, cur_loss = sess.run(
                    [minimize_sym, loss_sym],
                    feed_dict={
                        obs_ph: samples["observation"],
                        expert_action_ph: samples["action"]
                    })
                if not (num_steps % steps_print_loss):
                    print("Loss: {}".format(cur_loss))

                # To visualize how our clone model behaves, make a prediction
                # from the clone and step based on the retrieved action.
                action = sess.run(
                    action_ph, feed_dict={obs_ph: [obs.flatten()]})[0]
                obs, reward, done, info = env.step(action)
                num_steps += 1
    env.close()


if __name__ == '__main__':
    main()
