"""RL^2."""
import numpy as np

from garage.np.algos import RLAlgorithm
from dowel import logger, tabular
from garage.misc import tensor_utils as np_tensor_utils


class RL2(RLAlgorithm):
    def __init__(self, policy, inner_algo, max_path_length, normalize_adv=True, positive_adv=False):
        self._inner_algo = inner_algo
        self.max_path_length = max_path_length
        self.env_spec = inner_algo.env_spec
        self.flatten_input = inner_algo.flatten_input
        self.policy = inner_algo.policy
        self.baseline = inner_algo.baseline
        self.discount = inner_algo.discount
        self.gae_lambda = inner_algo.gae_lambda
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            runner.step_path = runner.obtain_samples(runner.step_itr)
            tabular.record('TotalEnvSteps', runner.total_env_steps)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Average return.

        """
        paths = self.process_samples(itr, paths)
        self._inner_algo.log_diagnostics(paths)
        logger.log('Optimizing policy...')
        self._inner_algo.optimize_policy(itr, paths)
        return paths['average_return']

    def process_samples(self, itr, paths):
        # pylint: disable=too-many-statements
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            dict: Processed sample data, with key
                * observations: (numpy.ndarray)
                * actions: (numpy.ndarray)
                * rewards: (numpy.ndarray)
                * baselines: (numpy.ndarray)
                * returns: (numpy.ndarray)
                * valids: (numpy.ndarray)
                * agent_infos: (dict)
                * env_infos: (dict)
                * paths: (list[dict])
                * average_return: (numpy.float64)

        """
        all_paths = []
        samples_data_meta_batch = []
        # Paths shape now: (meta_batch_size, (num_of_episode), *dims)
        # We want to reshape it into (meta_batch_size, num_of_episode * max_path_length, *dims)
        # We concatenate all episode, so we have (meta_batch_size, num_of_episode * max_path_length, *dims)
        for _, path in paths.items():
            samples_data, paths = self._compute_samples_data(path)
            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)

        observations, actions, rewards, baselines, returns, advantages, valids, env_infos, agent_infos = \
            self._stack_paths(samples_data_meta_batch)

        average_discounted_return = (np.mean(
            [path['returns'][0] for path in all_paths]))

        undiscounted_returns = [sum(path['rewards']) for path in all_paths]

        ent = np.sum(self.policy.distribution.entropy(agent_infos) *
                     valids) / np.sum(valids)

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            baselines=baselines,
            returns=returns,
            advantages=advantages,
            valids=valids,
            agent_infos=agent_infos,
            env_infos=env_infos,
            paths=all_paths,
            average_return=np.mean(undiscounted_returns),
        )

        tabular.record('Iteration', itr)
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('NumTrajs', len(all_paths))
        tabular.record('Entropy', ent)
        tabular.record('Perplexity', np.exp(ent))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))

        return samples_data

    def _compute_advantages_and_baselines(self, paths, all_path_baselines):
        assert len(paths) == len(all_path_baselines)
        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = np_tensor_utils.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path['baselines'] = all_path_baselines[idx]
        return paths

    def _concatenate_paths(self, paths):
        if self.flatten_input:
            observations = np.concatenate([self.env_spec.observation_space.flatten_n(
                path["observations"]) for path in paths])
        else:
            observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([self.env_spec.action_space.flatten_n(path["actions"])
            for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        dones = np.concatenate([path["dones"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        baselines = np.concatenate([path["baselines"] for path in paths])
        valids = np.concatenate([np.ones_like(path['returns']) for path in paths])
        env_infos = np_tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = np_tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        return observations, actions, rewards, dones, returns, advantages, baselines, valids, env_infos, agent_infos

    def _stack_paths(self, paths):
        max_total_path_length = self._inner_algo.max_path_length

        observations = self._stack_padding(paths, 'observations', max_total_path_length)
        actions = self._stack_padding(paths, 'actions', max_total_path_length)
        rewards = self._stack_padding(paths, 'rewards', max_total_path_length)
        returns = self._stack_padding(paths, 'returns', max_total_path_length)
        advantages = self._stack_padding(paths, 'advantages', max_total_path_length)
        baselines = self._stack_padding(paths, 'baselines', max_total_path_length)
        valids = self._stack_padding(paths, 'valids', max_total_path_length)
        returns = self._stack_padding(paths, 'returns', max_total_path_length)
        dones = self._stack_padding(paths, 'dones', max_total_path_length)
        env_infos = np_tensor_utils.stack_tensor_dict_list([path["env_infos"] for path in paths], max_total_path_length)
        agent_infos = np_tensor_utils.stack_tensor_dict_list([path["agent_infos"] for path in paths], max_total_path_length)

        return observations, actions, rewards, baselines, returns, advantages, valids, env_infos, agent_infos

    def _stack_padding(self, paths, key, max_path):
        padded_array = np.stack([
            np_tensor_utils.pad_tensor(path[key], max_path) for path in paths
        ])
        return padded_array

    def _compute_samples_data(self, paths):
        assert type(paths) == list
        for idx, path in enumerate(paths):
            path["returns"] = np_tensor_utils.discount_cumsum(path["rewards"], self.discount)
        # 2) fit baseline estimator using the path returns and predict the return baselines
        all_path_baselines = [self.baseline.predict(path) for path in paths]
        self.baseline.fit(paths)
        # 3) compute advantages and adjusted rewards
        paths = self._compute_advantages_and_baselines(paths, all_path_baselines)
        # 4) stack path data
        observations, actions, rewards, dones, returns, advantages, baselines, valids, env_infos, agent_infos = self._concatenate_paths(paths)
        # TODO(move advantage calculation from npo to here)
        # if desired normalize
        if self.normalize_adv:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        if self.positive_adv:
            advantages = (advantages - np.min(advantages)) + 1e-8
        # 6) create samples_data object
        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            returns=returns,
            advantages=advantages,
            baselines=baselines,
            valids=valids,
            env_infos=env_infos,
            agent_infos=agent_infos,
        )
        return samples_data, paths

