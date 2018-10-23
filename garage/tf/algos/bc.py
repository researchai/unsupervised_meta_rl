import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.algos import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class BC(OffPolicyRLAlgorithm):
    def __init__(self,
                 policy_lr=1e-4,
                 policy_optimizer=tf.train.AdamOptimizer,
                 name=None,
                 **kwargs):
        self.name = name
        self.policy_lr = policy_lr
        self.policy_optimizer = policy_optimizer

        super(BC, self).__init__(**kwargs)

    @overrides
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
                # TODO: Does MSE work for BC?
                action_loss = tf.reduce_mean(
                    tf.square(actions - expert_actions))

            with tf.name_scope("minimize_action_loss"):
                policy_train_op = self.policy_optimizer(
                    self.policy_lr, name="PolicyOptimizer").minimize(
                        action_loss, var_list=self.policy.get_trainable_vars())

            f_train_policy = tensor_utils.compile_function(
                inputs=[actions, expert_actions],
                outputs=[policy_train_op, action_loss])

            self.f_train_policy = f_train_policy

    @overrides
    def optimize_policy(self, itr, samples_data):
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        expert_actions = transitions["action"]

        actions = samples_data["actions"]

        _, action_loss = self.f_train_policy(actions, expert_actions)

        return action_loss

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
        )

    @overrides
    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        self.start_worker(sess)

        if self.use_target:
            self.f_init_target()

        episode_rewards = []
        episode_policy_losses = []
        last_average_return = None

        for epoch in range(self.n_epochs):
            with logger.prefix('epoch #%d | ' % epoch):
                for epoch_cycle in range(self.n_epoch_cycles):
                    # TODO: Couple sampling of policy to expert observations
                    paths = self.obtain_samples(epoch)
                    samples_data = self.process_samples(epoch, paths)
                    episode_rewards.extend(
                        samples_data["undiscounted_returns"])
                    self.log_diagnostics(paths)
                    for train_itr in range(self.n_train_steps):
                        if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:
                            self.evaluate = True
                            policy_loss = self.optimize_policy(
                                epoch, samples_data)

                            episode_policy_losses.append(policy_loss)

                    if self.plot:
                        self.plotter.update_plot(self.policy,
                                                 self.max_path_length)
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")

                logger.log("Training finished")
                logger.log("Saving snapshot #{}".format(epoch))
                params = self.get_itr_snapshot(epoch, samples_data)
                logger.save_itr_params(epoch, params)
                logger.log("Saved")
                if self.evaluate:
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
