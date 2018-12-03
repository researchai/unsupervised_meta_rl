"""
This module implements a BC model and associated classes.

BC (Behavioral Cloning) is a simple form of imitation learning that
infers a target policy from a fixed dataset of expert demonstations.
"""
import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.misc.overrides import overrides
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class BC(OffPolicyRLAlgorithm):
    """
    A Behavioral Cloning algorithm based on:
    Pomerleau, Dean A. "Alvinn: An autonomous land vehicle in a neural network."
    Advances in neural information processing systems. 1989.

    Example:
        $ python garage/examples/tf/bc_pointenv.py
    """

    def __init__(self,
                 env,
                 policy,
                 expert_dataset,
                 stochastic_policy=False,
                 policy_optimizer=tf.train.AdamOptimizer,
                 policy_lr=1e-3,
                 expert_batch_size=128,
                 name=None,
                 _no_train=False,
                 **kwargs):
        """
        Construct class.

        Args:
            env(): Environment
            policy(): Target policy network to learn expert behavior.
            expert_dataset(): Dataset of expert actions and observations.
            stochastic_policy(bool): Flag to indicate target policy is stochastic.
            policy_optimizer(): Optimizer for training target policy.
            policy_lr(float): Learning rate for training target policy.
            expert_batch_size(int): Batch size for each training step.
            name(str): Name of the algorithm .
            _no_train(bool): Flag for BCExpertEvaluator.
        """
        self.expert_dataset = expert_dataset
        self.stochastic_policy = stochastic_policy
        self.policy_lr = policy_lr
        self.policy_optimizer = policy_optimizer
        self.expert_batch_size = expert_batch_size
        self.name = name
        self._no_train = _no_train

        # We don't need use this for BC, but OffPolVecSampler expects a ReplayBuffer
        ph_replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)

        super(BC, self).__init__(
            env=env,
            policy=policy,
            qf=None,
            replay_buffer=ph_replay_buffer,
            **kwargs)

    @overrides
    def init_opt(self):
        """Build graph for BC training."""
        # Skip training graph construction if flag is set. This allows
        # BCExpertEvaluator to load a trained expert policy without error.
        if self._no_train:
            return

        with tf.name_scope(self.name, "BC"):
            # define input placeholders
            with tf.name_scope("inputs"):
                expert_actions = tf.placeholder(
                    tf.float32,
                    shape=(None, self.env.action_space.flat_dim),
                    name="input_expert_action")
                observations = tf.placeholder(
                    tf.float32,
                    shape=(None, self.env.observation_space.flat_dim),
                    name="input_observation")

            # define target policy action
            if self.stochastic_policy:
                target_dist = self.policy.dist_info_sym(
                    observations, name="target_action")
                target_actions = target_dist['mean'] + \
                    target_dist['log_std'] * tf.random_normal(tf.shape(target_dist['mean']))
            else:
                target_actions = self.policy.get_action_sym(
                    observations, name="target_action")

            # loss objective is MSE between expert and target actions
            with tf.name_scope("action_loss"):
                action_loss = tf.reduce_mean(
                    tf.losses.mean_squared_error(
                        predictions=target_actions, labels=expert_actions))

            # define minimizer function
            with tf.name_scope("minimize_action_loss"):
                policy_train_op = self.policy_optimizer(
                    self.policy_lr,
                    name="PolicyOptimizer").minimize(action_loss)

            f_train_policy = tensor_utils.compile_function(
                inputs=[observations, expert_actions],
                outputs=[policy_train_op, action_loss])

            self.f_train_policy = f_train_policy

    @overrides
    def optimize_policy(self, itr, samples_data):
        """
        Run BC optimization for a single batch.

        Args:
            itr(int): epoch number.
            samples_data(dict): batch of expert actions and observations.

        Returns:
            action_loss: Loss of action from target policy.
        """
        expert_actions = samples_data["expert_actions"]
        observations = samples_data["observations"]

        _, action_loss = self.f_train_policy(observations, expert_actions)
        return action_loss

    def gen_expert_sampler(self):
        """
        Generator for batch sampling expert dataset without replacement.

        Returns:
            batch_data(dict): batch of expert actions and observations.
        """
        dataset_length = self.expert_dataset["action"].shape[0]
        batch_idx = np.arange(dataset_length)
        while True:
            np.random.shuffle(batch_idx)
            for batchnum in range(dataset_length // self.expert_batch_size):
                batch = batch_idx[self.expert_batch_size * batchnum:
                                  self.expert_batch_size * (batchnum + 1)]
                exp_act = self.expert_dataset["action"][batch, :]
                exp_obs = self.expert_dataset["observation"][batch, :]

                assert exp_act.shape[0] == exp_obs.shape[0]
                assert exp_act.shape[0] == self.expert_batch_size

                batch_data = dict(observations=exp_obs, expert_actions=exp_act)
                yield batch_data

    @overrides
    def get_itr_snapshot(self, itr, expert_data, target_data):
        return dict(itr=itr, policy=self.policy, env=self.env)

    @overrides
    def log_diagnostics(self, paths):
        # Broken for GaussianMLPPolicy + OffPolicyVecSampler
        # self.policy.log_diagnostics(paths)
        pass

    @overrides
    def train(self, sess=None):
        """Run BC algorithm."""
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()
            sess.run(tf.global_variables_initializer())
        self.start_worker(sess)

        episode_rewards = []
        episode_policy_losses = []
        expert_sampler = self.gen_expert_sampler()
        last_average_return = None

        for epoch in range(self.n_epochs):
            with logger.prefix('epoch #%d | ' % epoch):
                for epoch_cycle in range(self.n_epoch_cycles):
                    expert_data = expert_sampler.__next__()

                    for train_itr in range(self.n_train_steps):
                        policy_loss = self.optimize_policy(epoch, expert_data)
                        episode_policy_losses.append(policy_loss)

                # parallel sampling used to collect reward data on target policy
                target_paths = self.obtain_samples(epoch)
                target_data = self.process_samples(epoch, target_paths)

                episode_rewards.extend(target_data["undiscounted_returns"])
                self.log_diagnostics(target_paths)

                logger.log("Training finished")
                logger.log("Saving snapshot #{}".format(epoch))
                params = self.get_itr_snapshot(epoch, expert_data, target_data)
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
                if self.plot:
                    self.plotter.update_plot(self.policy)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")

        self.shutdown_worker()
        if created_session:
            sess.close()
        return last_average_return


class BCExpertEvaluator(BC):
    """
    Evaluator class to measure the performance of a trained policy.
    Can be used to measure an expert policy (if it exists) for comparison against
    a BC-trained target policy.
    """

    def __init__(self, env, expert_policy, expert_tf_session, **kwargs):
        """
        Construct class.

        Args:
            env(): Environment.
            expert_policy(): Expert policy network. Graph must match saved model exactly.
            expert_tf_session(str): Path to tf checkpoint of expert policy network.
        """
        self.expert_tf_session = expert_tf_session

        # We set _no_train here to avoid adding new tf objects to the graph.
        # See: https://github.com/tensorflow/tensorflow/issues/17257
        super(BCExpertEvaluator, self).__init__(
            env=env,
            policy=expert_policy,
            expert_dataset=None,
            _no_train=True,
            **kwargs)

    @overrides
    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.expert_tf_session)
            self.start_worker(sess)

            episode_rewards = []
            last_average_return = None

            for epoch in range(self.n_epochs):
                with logger.prefix('(Expert Policy) epoch #%d | ' % epoch):
                    paths = self.obtain_samples(epoch)
                    samples_data = self.process_samples(epoch, paths)
                    episode_rewards.extend(
                        samples_data["undiscounted_returns"])

                    logger.log("Evaluation finished")
                    last_average_return = np.mean(episode_rewards)
                    logger.record_tabular('AverageReturn', last_average_return)
                    logger.record_tabular('StdReturn', np.std(episode_rewards))
                    logger.dump_tabular(with_prefix=False)

            self.shutdown_worker()
        return last_average_return
