from baselines import deepq
import gym

import tensorflow as tf

from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.envs import TfEnv


def sample_from_expert(env, size_in_transitions=600, time_horizon=200):
    """Sample actions and observations from the expert.

    We load the TensorFlow variables that define our expert from the file
    cartpole_model.pk, and sample actions and observations from it, saving
    the results in a replay buffer.

    Parameters:
        - env: the gym environment
        - buffer_size: number of samples that our replay buffer will contain
    Returns:
        - replay_buffer: a dictionary with arrays of observations and actions
          with size defined by buffer_size.
    """
    act = deepq.load("garage/tf/behavioral_cloning/cartpole_model.pkl")
    replay_buffer = SimpleReplayBuffer(
        env_spec=env.spec,
        size_in_transitions=size_in_transitions,
        time_horizon=time_horizon)

    while not replay_buffer.full:
        obs, done = env.reset(), False
        samples = 0
        while not done and (not replay_buffer.full):
            action = act(obs[None])[0]
            buffer_action = [0, 1]
            if not action:
                buffer_action = [1, 0]
            next_obs, rew, done, _ = env.step(action)
            samples += 1
            replay_buffer.add_transition(
                action=[buffer_action],
                observation=[obs],
                terminal=[done],
                reward=[rew],
                next_observation=[next_obs])
            obs = next_obs
        print("Samples collected by episode in the expert", samples)
    return replay_buffer


def build_nn(env):
    """Build a neural network.

    This neural network represents the clone of our expert, where the
    observations of the environment are the input and the actions are the
    output.

    Parameters:
        - env: the gym environment.

    Returns:
        - obs_ph: the observation placeholder is the input of neural network.
        - action_sym: symbolic operation to retrieve the action (output) from
          the neural network.
        - logits_sym: symbolic operation to retrieve the logits values for
          each possible action in the environment. This is useful to calculate
          the loss.
    """
    obs_dimension = [None] + list(env.observation_space.shape)
    obs_ph = tf.placeholder(tf.float32, shape=obs_dimension)

    with tf.variable_scope("layer1"):
        hidden = tf.layers.dense(obs_ph, 128, activation=tf.nn.relu)
    with tf.variable_scope("layer2"):
        hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu)
    with tf.variable_scope("layer3"):
        logits_sym = tf.layers.dense(hidden, env.action_space.n)
    with tf.variable_scope("output"):
        action_sym = tf.argmax(input=logits_sym, axis=1)

    return obs_ph, action_sym, logits_sym


def optimize_nn(num_actions, predicted_action_logits):
    """Build the optimizer for the clone of our expert.

    It reduces the loss between the logits output of the clone and the one-hot
    representation of the actions provided by the expert in the buffer.

    Parameters:
        - num_actions: the number of possible actions that can be taken in
          the environment. This is used to build a one-hot representation
          of the actions in the buffer.
    Returns:
        - minimize_sym: obtain the symbolic operation to minimize the loss in
          our clone with respect the samples from the expert.
        - loss_sym: obtain the symbolic operation for the loss, which is useful
          to visualize how the loss is minimized.
        - expert_action_ph: the placeholder for the expert actions. This is one
          of the two inputs to calculate the loss in our optimizer.
    """
    expert_action_ph = tf.placeholder(tf.int32, shape=[None, 2])

    with tf.variable_scope("loss"):
        loss_sym = tf.losses.softmax_cross_entropy(
            onehot_labels=expert_action_ph, logits=predicted_action_logits)
        loss_sym = tf.reduce_mean(loss_sym)

    with tf.variable_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        minimize_sym = optimizer.minimize(loss=loss_sym)

    return minimize_sym, loss_sym, expert_action_ph


def main():
    env_id = "CartPole-v0"
    env = TfEnv(gym.make(env_id))
    replay_buffer = sample_from_expert(env)
    obs_ph, action_sym, logits_sym = build_nn(env)
    minimize_sym, loss_sym, expert_action_ph = (optimize_nn(
        env.action_space.n, logits_sym))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    steps_print_loss = 100
    num_steps = 0
    while True:
        done = False
        obs = env.reset()
        while not done:
            env.render()
            # Get a random batch from the data
            samples = replay_buffer.sample(128)

            # Train the model.
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
                action_sym, feed_dict={obs_ph: [obs.flatten()]})[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1
    env.close()


if __name__ == '__main__':
    main()
