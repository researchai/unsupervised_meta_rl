import time

import numpy as np
import tensorflow as tf

from garage.algos import RLAlgorithm
from garage.misc import special, tensor_utils
import garage.misc.logger as logger
from garage.tf.plotter import Plotter
from garage.tf.samplers import BatchSampler
from garage.tf.samplers import OnPolicyVectorizedSampler


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo,
    etc.
    """

    def __init__(self,
                 env,
                 policy,
                 baseline,
                 scope=None,
                 n_itr=500,
                 start_itr=0,
                 batch_size=5000,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 plot=False,
                 pause_for_plot=False,
                 center_adv=True,
                 positive_adv=False,
                 store_paths=False,
                 whole_paths=True,
                 fixed_horizon=False,
                 sampler_cls=None,
                 sampler_args=None,
                 force_batch_sampler=False,
                 **kwargs):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if
         running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have
         mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are
         always positive. When used in conjunction with center_adv the
         advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = OnPolicyVectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

    def start_worker(self, sess):
        self.sampler.start_worker()
        if self.plot:
            self.plotter = Plotter(self.env, self.policy, sess)
            self.plotter.start()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()
        if self.plot:
            self.plotter.close()

    def get_actions(self, obses):
        return self.policy.get_actions(obses)

    def obtain_samples(self):
        obs = self.sampler.reset()
        while self.sampler.sample_count < self.batch_size:
            actions = self.policy.get_actions(obs)
            obs = self.sampler.step(actions)
        
        return self.sampler.get_samples()

    def process_samples(self, paths):
        baselines = []
        returns = []

        max_path_length = self.max_path_length

        if hasattr(self.baseline, "predict_n"):
            all_path_baselines = self.baseline.predict_n(paths)
        else:
            all_path_baselines = [
                self.baseline.predict(path) for path in paths
            ]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                self.discount * path_baselines[1:] - \
                path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["deltas"] = deltas

        for idx, path in enumerate(paths):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])

            # returns
            path["returns"] = special.discount_cumsum(path["rewards"],
                                                      self.discount)
            returns.append(path["returns"])

        # make all paths the same length
        obs = [path["observations"] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path["actions"] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        rewards = [path["rewards"] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        returns = [path["returns"] for path in paths]
        returns = tensor_utils.pad_tensor_n(returns, max_path_length)

        advantages = [path["advantages"] for path in paths]
        advantages = tensor_utils.pad_tensor_n(advantages, max_path_length)

        baselines = tensor_utils.pad_tensor_n(baselines, max_path_length)

        agent_infos = [path["agent_infos"] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        env_infos = [path["env_infos"] for path in paths]
        env_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos
        ])

        valids = [np.ones_like(path["returns"]) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        average_discounted_return = (np.mean(
            [path["returns"][0] for path in paths]))

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        ent = np.sum(
            self.policy.distribution.entropy(agent_infos) *
            valids) / np.sum(valids)

        samples_data = dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
            baselines=baselines,
            returns=returns,
            valids=valids,
            agent_infos=agent_infos,
            env_infos=env_infos,
            paths=paths,
            average_return=np.mean(undiscounted_returns),
        )

        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data

    def train_once(self, paths, sess=None):
        logger.log("Processing samples...")
        processed_paths = self.process_samples(paths)
        last_average_return = processed_paths["average_return"]
        logger.log("Logging diagnostics...")
        self.log_diagnostics(paths)

        logger.log("Optimizing policy...")
        self.optimize_policy(processed_paths)

        if self.plot:
            self.plotter.update_plot(self.policy, self.max_path_length)
            if self.pause_for_plot:
                input("Plotting evaluation run: Press Enter to " "continue...")

        return last_average_return

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        self.start_worker(sess)
        start_time = time.time()
        last_average_return = None
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)
                last_average_return = self.train_once(paths)

                # TODO: support snapshotor
                # logger.log("Saving snapshot...")
                # params = self.get_itr_snapshot(itr, samples_data)
                # if self.store_paths:
                #     params["paths"] = samples_data["paths"]
                # logger.save_itr_params(itr, params)
                # logger.log("Saved")

                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.plotter.update_plot(self.policy, self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")

        self.shutdown_worker()
        if created_session:
            sess.close()
        return last_average_return

    def log_diagnostics(self, paths):
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, samples_data):
        raise NotImplementedError
