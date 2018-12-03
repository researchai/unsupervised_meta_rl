import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.misc.overrides import overrides
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class BC(OffPolicyRLAlgorithm):
    def __init__(self,
                 env,
                 policy,
                 expert_dataset,
                 stochastic_policy=False,
                 policy_optimizer=tf.train.AdamOptimizer,
                 policy_lr=1e-3,
                 expert_batch_size=128,
                 name=None,
                 no_train=False,
                 **kwargs):
        self.expert_dataset = expert_dataset
        self.stochastic_policy = stochastic_policy
        self.policy_lr = policy_lr
        self.policy_optimizer = policy_optimizer
        self.expert_batch_size = expert_batch_size
        self.name = name
        self.no_train = no_train

        # We don't need this for BC, but OffPolVecSampler expects this
        dummy_replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)

        super(BC, self).__init__(
            env=env,
            policy=policy,
            qf=None,
            replay_buffer=dummy_replay_buffer,
            **kwargs)

    @overrides
    def init_opt(self):
        if self.no_train:
            return

        with tf.name_scope(self.name, "BC"):

            with tf.name_scope("inputs"):
                expert_actions = tf.placeholder(
                    tf.float32,
                    shape=(None, self.env.action_space.flat_dim),
                    name="input_expert_action")
                observations = tf.placeholder(
                    tf.float32,
                    shape=(None, self.env.observation_space.flat_dim),
                    name="input_observation")

            if self.stochastic_policy:
                target_dist = self.policy.dist_info_sym(
                    observations, name="target_action")
                target_actions = target_dist['mean'] + \
                    target_dist['log_std'] * tf.random_normal(tf.shape(target_dist['mean']))
            else:
                target_actions = self.policy.get_action_sym(
                    observations, name="target_action")

            with tf.name_scope("action_loss"):
                action_loss = tf.reduce_mean(
                    tf.losses.mean_squared_error(
                        predictions=target_actions, labels=expert_actions))

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
        expert_actions = samples_data["expert_actions"]
        observations = samples_data["observations"]

        _, action_loss = self.f_train_policy(observations, expert_actions)

        return action_loss

    def gen_expert_sampler(self):
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

                yield dict(observations=exp_obs, expert_actions=exp_act)

    @overrides
    def get_itr_snapshot(self, itr, expert_data, target_data):
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
        )

    @overrides
    def log_diagnostics(self, paths):
        # Broken for GaussianMLPPolicy + OffPolicyVecSampler
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
        expert_sampler = self.gen_expert_sampler()
        last_average_return = None

        for epoch in range(self.n_epochs):
            with logger.prefix('epoch #%d | ' % epoch):
                for epoch_cycle in range(self.n_epoch_cycles):
                    expert_data = expert_sampler.__next__()

                    for train_itr in range(self.n_train_steps):
                        policy_loss = self.optimize_policy(epoch, expert_data)
                        episode_policy_losses.append(policy_loss)

                    if self.plot:
                        self.plotter.update_plot(self.policy)
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")

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

        self.shutdown_worker()
        if created_session:
            sess.close()
        return last_average_return


class BCExpertEvaluator(BC):
    def __init__(self, env, policy, expert_tf_session, **kwargs):
        self.expert_tf_session = expert_tf_session

        super(BCExpertEvaluator, self).__init__(
            env=env,
            policy=policy,
            expert_dataset=None,
            no_train=True,
            **kwargs)

    @overrides
    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.expert_tf_session)
            self.start_worker(sess)

            episode_rewards = []
            last_average_return = None

            with logger.prefix('(Expert Policy) | '):
                for epoch in range(self.n_epochs):
                    for epoch_cycle in range(self.n_epoch_cycles):
                        paths = self.obtain_samples(0)
                        samples_data = self.process_samples(0, paths)
                        episode_rewards.extend(
                            samples_data["undiscounted_returns"])

                    logger.record_tabular('AverageReturn',
                                          np.mean(episode_rewards))
                    logger.record_tabular('StdReturn', np.std(episode_rewards))
                    last_average_return = np.mean(episode_rewards)
                    logger.dump_tabular(with_prefix=False)

            self.shutdown_worker()
        return last_average_return


def sample_from_expert(env,
                       expert_policy,
                       expert_tf_session,
                       dataset_save_path,
                       size_in_transitions=800,
                       time_horizon=200):
    replay_buffer = SimpleReplayBuffer(
        env_spec=env.spec,
        size_in_transitions=size_in_transitions,
        time_horizon=time_horizon)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, expert_tf_session)
        while not replay_buffer.full:
            done = False
            obs = env.reset()
            samples = 0
            while not done and (not replay_buffer.full):
                action, _ = expert_policy.get_action(obs)
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
    np.savez(dataset_save_path, **data)
    # Retrieve data as:
    # expert = np.load("dataset_save_path")
    # obs = expert["observation"]
    # acts = expert["action"]

    return data
