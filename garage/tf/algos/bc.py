import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.misc.overrides import overrides
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import RLAlgorithm
from garage.tf.misc import tensor_utils
from garage.tf.plotter import Plotter


class BC(RLAlgorithm):
    def __init__(self,
                 env,
                 policy,
                 expert_dataset,
                 policy_lr=1e-3,
                 policy_optimizer=tf.train.AdamOptimizer,
                 plot=False,
                 pause_for_plot=False,
                 smooth_return=True,
                 n_epochs=500,
                 n_epoch_cycles=20,
                 n_train_steps=50,
                 name=None,
                 **kwargs):
        self.env = env
        self.policy = policy
        self.expert_dataset = expert_dataset
        self.policy_lr = policy_lr
        self.policy_optimizer = policy_optimizer
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.smooth_return = smooth_return
        self.n_epochs = n_epochs
        self.n_epoch_cycles = n_epoch_cycles
        self.n_train_steps = n_train_steps
        self.name = name

        self.init_opt()

    def start_worker(self, sess):
        """Initialize sampler and plotter."""
        self.sampler.start_worker()
        if self.plot:
            self.plotter = Plotter(self.env, self.policy, sess)
            self.plotter.start()

    def shutdown_worker(self):
        """Close sampler and plotter."""
        self.sampler.shutdown_worker()
        if self.plot:
            self.plotter.close()

    def init_opt(self):
        with tf.name_scope(self.name, "BC"):

            with tf.name_scope("inputs"):
                actions = tf.placeholder(
                    tf.float32,
                    shape=(None, self.env.action_space.flat_dim),
                    name="input_action")
                expert_actions = tf.placeholder(
                    tf.float32,
                    shape=(None, self.env.action_space.flat_dim),
                    name="input_expert_action")

            with tf.name_scope("action_loss"):
                action_loss = tf.reduce_mean(
                    tf.squared_difference(actions, expert_actions))

            with tf.name_scope("minimize_action_loss"):
                policy_train_op = self.policy_optimizer(
                    self.policy_lr, name="PolicyOptimizer").minimize(
                        action_loss, var_list=self.policy.get_trainable_vars())

            f_train_policy = tensor_utils.compile_function(
                inputs=[actions, expert_actions],
                outputs=[policy_train_op, action_loss])

            self.f_train_policy = f_train_policy

    def optimize_policy(self, itr, samples_data):
        expert_actions = samples_data["expert_actions"]
        target_actions = samples_data["target_actions"]

        _, action_loss = self.f_train_policy(target_actions, expert_actions)

        return action_loss

    def obtain_samples(self, itr):
        """Select a batch of (obs,act) pairs from the expert. Then sample target with obs"""
        expert_batch = self.expert_dataset.sample(self.buffer_batch_size)
        assert "action" in expert_batch
        assert "observation" in expert_batch

        # TODO: Calculate expert and target actions' reward
        if self.policy.vectorized:
            target_actions, _ = self.policy.get_actions(
                expert_batch["observation"])
        else:
            target_actions = [
                self.policy.get_action(exp_obs)
                for exp_obs in expert_batch["observation"]
            ]
            target_actions = np.array(target_actions)

        return dict(
            observations=expert_batch["observation"],
            expert_actions=expert_batch["action"],
            target_actions=target_actions,
        )

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
        )

    def log_diagnostics(self, paths):
        """ We break too many APIs for this to work! """
        # self.policy.log_diagnostics(paths)
        pass

    @overrides
    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        self.start_worker(sess)

        episode_rewards = []
        episode_policy_losses = []
        last_average_return = None

        for epoch in range(self.n_epochs):
            with logger.prefix('epoch #%d | ' % epoch):
                for epoch_cycle in range(self.n_epoch_cycles):
                    samples_data = self.obtain_samples(epoch)
                    episode_rewards.extend(
                        samples_data["undiscounted_returns"])
                    self.log_diagnostics(samples_data)
                    for train_itr in range(self.n_train_steps):
                        policy_loss = self.optimize_policy(epoch, samples_data)
                        episode_policy_losses.append(policy_loss)

                    if self.plot:
                        self.plotter.update_plot(self.policy)
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")

                logger.log("Training finished")
                logger.log("Saving snapshot #{}".format(epoch))
                params = self.get_itr_snapshot(epoch, samples_data)
                logger.save_itr_params(epoch, params)
                logger.log("Saved")

                logger.record_tabular('Epoch', epoch)
                logger.record_tabular('AverageReturn',
                                      np.mean(episode_rewards))
                logger.record_tabular('StdReturn', np.std(episode_rewards))
                logger.record_tabular('Policy/AveragePolicyLoss',
                                      np.mean(episode_policy_losses))
                last_average_return = np.mean(episode_rewards)

                if not self.smooth_return:
                    episode_rewards = []
                    episode_policy_losses = []

                logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()
        if created_session:
            sess.close()
        return last_average_return


def sample_expert_policy_to_dataset(env,
                                    policy,
                                    tf_session_pkl,
                                    size_in_transitions=800,
                                    time_horizon=200):
    buffer_shapes = {
        "action": env.action_space.shape,
        "observation": env.observation_space.shape
    }
    replay_buffer = SimpleReplayBuffer(
        env_spec=env.spec,
        size_in_transitions=size_in_transitions,
        time_horizon=time_horizon)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        expert_tf_session = (tf_session_pkl)
        saver.restore(sess, expert_tf_session)
        while not replay_buffer.full:
            done = False
            obs = env.reset()
            samples = 0
            while not done and (not replay_buffer.full):
                action, _ = policy.get_action(obs)
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
