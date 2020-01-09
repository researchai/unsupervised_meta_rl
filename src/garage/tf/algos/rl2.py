"""RL^2."""
import numpy as np

from garage.np.algos import RLAlgorithm
from dowel import logger, tabular
from garage.misc import tensor_utils as np_tensor_utils
from garage.tf.misc import tensor_utils


class RL2(RLAlgorithm):
    def __init__(self, policy, inner_algo, max_path_length, episode_per_task, normalize_adv=True, positive_adv=False):
        self._inner_algo = inner_algo
        self.max_path_length = max_path_length
        self.env_spec = inner_algo.env_spec
        self.flatten_input = inner_algo.flatten_input
        self.policy = inner_algo.policy
        self.baselines = inner_algo.baselines
        self.discount = inner_algo.discount
        self.episode_per_task = episode_per_task
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
        lengths = []
        # Paths shape now: (meta_batch_size, (num_of_episode), *dims)
        # We want to reshape it into (meta_batch_size, num_of_episode * max_path_length, *dims)
        # We concatenate all episode, so we have (meta_batch_size, num_of_episode * max_path_length, *dims)
        for idx, path in paths.items():
            samples_data, paths = self._compute_samples_data(idx, path)
            #samples_data contain all the paths in one meta batch, shape: [max_path_length * episoder_per_task, *dims]
            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)

        observations, actions, rewards, terminals, baselines, returns, advantages, mean_adv, deltas, valids, lengths, env_infos, agent_infos = \
            self._stack_paths(samples_data_meta_batch)

        _observations, _actions, _rewards, _terminals, _lengths, _env_infos, _agent_infos = \
            self._stack_paths_for_evaluation(all_paths)

        average_discounted_return = (np.mean(
            [path['returns'][0] for path in all_paths]))

        undiscounted_returns = [sum(path['rewards']) for path in all_paths]

        ent = np.sum(self.policy.distribution.entropy(agent_infos) *
                     valids) / np.sum(valids)

        undiscounted_returns = self.evaluate_performance(
            itr,
            dict(env_spec=self.env_spec,
                 observations=_observations,
                 actions=_actions,
                 rewards=_rewards,
                 terminals=_terminals,
                 env_infos=_env_infos,
                 agent_infos=_agent_infos,
                 lengths=_lengths,
                 discount=self.discount,
                 episode_reward_mean=self._inner_algo.episode_reward_mean))

        self._inner_algo.episode_reward_mean.extend(undiscounted_returns)

        tabular.record('Entropy', ent)
        tabular.record('Perplexity', np.exp(ent))
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self._inner_algo.episode_reward_mean))

        # all paths in each meta batch is stacked, shape: [meta_batch, max_path_length * episoder_per_task, *dims]
        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            baselines=baselines,
            valids=valids,
            lengths=lengths,
            agent_infos=agent_infos,
            env_infos=env_infos,
            paths=samples_data_meta_batch,
            average_return=np.mean(undiscounted_returns),
            # for debug ,remove later
            returns=returns,
            advantages=advantages,
            mean_adv=mean_adv,
            deltas=deltas
        )

        return samples_data

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
        deltas = np.concatenate([path["deltas"] for path in paths])
        valids = np.concatenate([np.ones_like(path['returns']) for path in paths])
        env_infos = np_tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = np_tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
        lengths = np.cumsum([len(path['rewards']) for path in paths])

        return observations, actions, rewards, dones, returns, advantages, deltas, baselines, valids, lengths, env_infos, agent_infos

    def _stack_paths(self, paths):
        max_total_path_length = self._inner_algo.max_path_length

        observations = np_tensor_utils.stack_and_pad_tensor_n(paths, 'observations', max_total_path_length)
        actions = np_tensor_utils.stack_and_pad_tensor_n(paths, 'actions', max_total_path_length)
        rewards = np_tensor_utils.stack_and_pad_tensor_n(paths, 'rewards', max_total_path_length)
        dones = np_tensor_utils.stack_and_pad_tensor_n(paths, 'dones', max_total_path_length)

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = np_tensor_utils.pad_tensor_n(valids, max_total_path_length)

        lengths = np.stack([path['lengths'] for path in paths])

        agent_infos = np_tensor_utils.stack_and_pad_tensor_dict(paths, 'agent_infos', max_total_path_length)
        env_infos = np_tensor_utils.stack_and_pad_tensor_dict(paths, 'env_infos', max_total_path_length)

        # for debug, delete later
        returns = np_tensor_utils.stack_and_pad_tensor_n(paths, 'returns', max_total_path_length)
        advantages = np_tensor_utils.stack_and_pad_tensor_n(paths, 'advantages', max_total_path_length)
        baselines = np_tensor_utils.stack_and_pad_tensor_n(paths, 'baselines', max_total_path_length)
        deltas = np_tensor_utils.stack_and_pad_tensor_n(paths, 'deltas', max_total_path_length)

        mean_adv = np.stack([path['mean_adv'] for path in paths])

        return observations, actions, rewards, dones, baselines, returns, advantages, mean_adv, deltas, valids, lengths, env_infos, agent_infos

    def _stack_paths_for_evaluation(self, paths):
        observations = np_tensor_utils.stack_and_pad_tensor_n(paths, 'observations', self.max_path_length)
        actions = np_tensor_utils.stack_and_pad_tensor_n(paths, 'actions', self.max_path_length)
        rewards = np_tensor_utils.stack_and_pad_tensor_n(paths, 'rewards', self.max_path_length)
        dones = np_tensor_utils.stack_and_pad_tensor_n(paths, 'dones', self.max_path_length)
        agent_infos = np_tensor_utils.stack_and_pad_tensor_dict(paths, 'agent_infos', self.max_path_length)
        env_infos = np_tensor_utils.stack_and_pad_tensor_dict(paths, 'env_infos', self.max_path_length)

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = np_tensor_utils.pad_tensor_n(valids, self.max_path_length)

        lengths = np.asarray([v.sum() for v in valids])

        return observations, actions, rewards, dones, lengths, env_infos, agent_infos

    def _compute_samples_data(self, i, paths):
        assert type(paths) == list
        for idx, path in enumerate(paths):
            path["returns"] = np_tensor_utils.discount_cumsum(path["rewards"], self.discount)
        # ##################
        # for debug, delete later
        self.baselines[i].fit(paths)
        all_path_baselines = [self.baselines[i].predict(path) for path in paths]
        paths = self._compute_advantages_and_baselines(paths, all_path_baselines)
        # ##################

        observations, actions, rewards, dones, returns, advantages, deltas, baselines, valids, lengths, env_infos, agent_infos = self._concatenate_paths(paths)
        
        # ##################
        # for debug, delete later
        mean_adv = np.mean(advantages)
        if self.normalize_adv:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        if self.positive_adv:
            advantages = (advantages - np.min(advantages)) + 1e-8
        # ##################
        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            valids=valids,
            lengths=lengths,
            agent_infos=agent_infos,
            env_infos=env_infos,
            # for debug, delete later
            returns=returns,
            advantages=advantages,
            mean_adv=mean_adv,
            baselines=baselines,
            deltas=deltas,
        )
        return samples_data, paths

    def _compute_advantages_and_baselines(self, paths, all_path_baselines):
        assert len(paths) == len(all_path_baselines)
        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path['deltas'] = deltas
            path["advantages"] = np_tensor_utils.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path['baselines'] = all_path_baselines[idx]
        return paths