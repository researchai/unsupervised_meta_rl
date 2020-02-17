"""RL^2: Fast Reinforcement learning via slow reinforcement learning.

Implemented in TensorFlow.

Reference: https://arxiv.org/pdf/1611.02779.pdf.
"""
import collections

from dowel import logger, tabular
import numpy as np

import garage
from garage import log_performance
from garage import TrajectoryBatch
from garage.misc import tensor_utils as np_tensor_utils
from garage.np.algos import MetaRLAlgorithm


class RL2(MetaRLAlgorithm):
    """RL^2.

    Args:
        inner_algo (garage.np.algos.RLAlgorithm): Inner algorithm.
        max_path_length (int): Maximum length for trajectories with respect
            to RL^2. Notice that it is differen from the maximum path length
            for the inner algorithm.
        meta_batch_size (int): Meta batch size.
        task_sampler (garage.experiment.TaskSampler): Task sampler.

    """

    def __init__(self, *, inner_algo, max_path_length, meta_batch_size,
                 task_sampler):
        assert isinstance(inner_algo, garage.tf.algos.BatchPolopt)
        self._inner_algo = inner_algo
        self._max_path_length = max_path_length
        self._env_spec = inner_algo.env_spec
        self._flatten_input = inner_algo.flatten_input
        self._policy = inner_algo.policy
        self._discount = inner_algo.discount
        self._meta_batch_size = meta_batch_size
        self._task_sampler = task_sampler

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch.

        """
        last_return = None

        for _ in runner.step_epochs():
            runner.step_path = runner.obtain_samples(
                runner.step_itr,
                env_update=self._task_sampler.sample(self._meta_batch_size))
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
        paths = self._process_samples(itr, paths)
        logger.log('Optimizing policy...')
        self._inner_algo.optimize_policy(itr, paths)
        return paths['average_return']

    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            object: The policy used to obtain samples that are later
                used for meta-RL adaptation.

        """
        self._policy.reset()

        class NoResetPolicy:
            """A policy that does not reset.

            Args:
                policy (garage.tf.policies.base.Policy): Policy.

            Returns:
                object: A policy that does not reset.

            """

            def __init__(self, policy):
                self._policy = policy

            @property
            def _prev_hiddens(self):
                """Previous hidden state.

                Returns:
                    np.ndarray: Previous hidden state, with shape
                        :math:`(N, hidden_dim)`

                """
                return self._policy.prev_hiddens

            def reset(self, dones=None):
                """Reset the policy.

                Args:
                    dones (numpy.ndarray): Bool that indicates terminal
                        state(s).

                """

            def get_action(self, obs):
                """Get a single action from this policy.

                Args:
                    obs (numpy.ndarray): Observation from environment.

                Returns:
                    tuple[numpy.ndarray, dict]: Predicted action and
                        agent info.

                """
                return self._policy.get_action(obs)

            def get_param_values(self):
                """Get param values.

                Returns:
                    np.ndarray: Values of the parameters evaluated in
                        the current session

                """
                return self._policy.get_param_values()

            def set_param_values(self, params):
                """Set param values.

                Args:
                    params (np.ndarray): A numpy array of parameter values.

                """
                self._policy.set_param_values(params)

        return NoResetPolicy(self._policy)

    def adapt_policy(self, exploration_policy, exploration_trajectories):
        """Produce a policy adapted for a task.

        Args:
            exploration_policy (garage.Policy): A policy which was returned
                from get_exploration_policy(), and which generated
                exploration_trajectories by interacting with an environment.
                The caller may not use this object after passing it into this
                method.
            exploration_trajectories (garage.TrajectoryBatch): Trajectories to
                adapt to, generated by exploration_policy exploring the
                environment.

        Returns:
            garage.Policy: A policy adapted to the task represented by the
                exploration_trajectories.

        """
        return exploration_policy

    def _process_samples(self, itr, paths):
        # pylint: disable=too-many-statements
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (OrderedDict[dict]): A list of collected paths for each
                task. In RL^2, there are n environments/tasks and paths in
                each of them will be concatenated at some point and fed to
                the policy.

        Returns:
            dict: Processed sample data, with key
                * observations: (numpy.ndarray)
                * actions: (numpy.ndarray)
                * rewards: (numpy.ndarray)
                * returns: (numpy.ndarray)
                * valids: (numpy.ndarray)
                * agent_infos: (dict)
                * env_infos: (dict)
                * paths: (list[dict])
                * average_return: (numpy.float64)

        Raises:
            ValueError: If 'batch_idx' is not found.

        """
        concatenated_path_in_meta_batch = []
        lengths = []

        paths_by_task = collections.defaultdict(list)
        for path in paths:
            path['returns'] = np_tensor_utils.discount_cumsum(
                path['rewards'], self._discount)
            path['lengths'] = [len(path['rewards'])]
            if 'batch_idx' in path:
                paths_by_task[path['batch_idx']].append(path)
            elif 'batch_idx' in path['agent_infos']:
                paths_by_task[path['agent_infos']['batch_idx'][0]].append(path)
            else:
                raise ValueError('Batch idx is required for RL2 but not found')

        # all path in paths_by_task[i] are sampled from task[i]
        #
        for path in paths_by_task.values():
            concatenated_path = self._concatenate_paths(path)
            concatenated_path_in_meta_batch.append(concatenated_path)

        # prepare paths for inner algorithm
        # pad the concatenated paths
        (observations, actions, rewards, _, _, valids, lengths, env_infos,
            agent_infos) = \
            stack_paths(max_len=self._inner_algo.max_path_length,
                        paths=concatenated_path_in_meta_batch)

        # prepare paths for performance evaluation
        # performance is evaluated across all paths, so each path
        # is padded with self._max_path_length
        (_observations, _actions, _rewards, _terminals, _, _valids, _lengths,
            _env_infos, _agent_infos) = \
            stack_paths(max_len=self._max_path_length,
                        paths=paths)

        ent = np.sum(self._policy.distribution.entropy(agent_infos) *
                     valids) / np.sum(valids)

        undiscounted_returns = log_performance(
            itr, TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
            self._inner_algo.discount)

        tabular.record('Entropy', ent)
        tabular.record('Perplexity', np.exp(ent))

        # all paths in each meta batch is stacked together
        # shape: [meta_batch, max_path_length * episoder_per_task, *dims]
        # per RL^2
        concatenated_path = dict(observations=observations,
                                 actions=actions,
                                 rewards=rewards,
                                 valids=valids,
                                 lengths=lengths,
                                 baselines=np.zeros_like(rewards),
                                 agent_infos=agent_infos,
                                 env_infos=env_infos,
                                 paths=concatenated_path_in_meta_batch,
                                 average_return=np.mean(undiscounted_returns))

        return concatenated_path

    def _concatenate_paths(self, paths):
        """Concatenate paths.

        The input paths are from different rollouts but same task/environment.
        In RL^2, paths within each meta batch are all concatenate into a single
        path and fed to the policy.

        Args:
            paths (dict): Input paths. All paths are from different rollouts,
                but the same task/environment.

        Returns:
            dict: Concatenated paths from the same task/environment. Shape of
                values: :math:`[max_path_length * episode_per_task, S^*]`
            list[dict]: Original input paths. Length of the list is
                :math:`episode_per_task` and each path in the list has
                values of shape :math:`[max_path_length, S^*]`

        """
        returns = []

        if self._flatten_input:
            observations = np.concatenate([
                self._env_spec.observation_space.flatten_n(
                    path['observations']) for path in paths
            ])
        else:
            observations = np.concatenate(
                [path['observations'] for path in paths])
        actions = np.concatenate([
            self._env_spec.action_space.flatten_n(path['actions'])
            for path in paths
        ])
        rewards = np.concatenate([path['rewards'] for path in paths])
        dones = np.concatenate([path['dones'] for path in paths])
        valids = np.concatenate(
            [np.ones_like(path['rewards']) for path in paths])
        returns = np.concatenate([path['returns'] for path in paths])

        env_infos = np_tensor_utils.concat_tensor_dict_list(
            [path['env_infos'] for path in paths])
        agent_infos = np_tensor_utils.concat_tensor_dict_list(
            [path['agent_infos'] for path in paths])
        lengths = [path['lengths'] for path in paths]

        concatenated_path = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            valids=valids,
            lengths=lengths,
            returns=returns,
            agent_infos=agent_infos,
            env_infos=env_infos,
        )
        return concatenated_path

    @property
    def policy(self):
        """Policy.

        Returns:
            garage.Policy: Policy to be used.

        """
        return self._inner_algo.policy

    @property
    def max_path_length(self):
        """Max path length.

        Returns:
            int: Maximum path length in a trajectory.

        """
        return self._max_path_length


def stack_paths(max_len, paths):
    """Pad paths to max_len and stacked all paths together.

    Args:
        max_len (int): Maximum path length.
        paths (dict): Input paths. Each path represents the concatenated paths
            from each meta batch (environment/task).

    Returns:
        numpy.ndarray: Observations. Shape:
            :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
        numpy.ndarray: Actions. Shape:
            :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
        numpy.ndarray: Rewards. Shape:
            :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
        numpy.ndarray: Terminal signals. Shape:
            :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
        numpy.ndarray: Returns. Shape:
            :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
        numpy.ndarray: Valids. Shape:
            :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
        numpy.ndarray: Lengths. Shape:
            :math:`[meta_batch, episode_per_task]`
        dict[numpy.ndarray]: Environment Infos. Shape of values:
            :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
        dict[numpy.ndarray]: Agent Infos. Shape of values:
            :math:`[meta_batch, episode_per_task * max_path_length, S^*]`

    """
    observations = np_tensor_utils.stack_and_pad_tensor_n(
        paths, 'observations', max_len)
    actions = np_tensor_utils.stack_and_pad_tensor_n(paths, 'actions', max_len)
    rewards = np_tensor_utils.stack_and_pad_tensor_n(paths, 'rewards', max_len)
    dones = np_tensor_utils.stack_and_pad_tensor_n(paths, 'dones', max_len)
    returns = np_tensor_utils.stack_and_pad_tensor_n(paths, 'returns', max_len)
    agent_infos = np_tensor_utils.stack_and_pad_tensor_n(
        paths, 'agent_infos', max_len)
    env_infos = np_tensor_utils.stack_and_pad_tensor_n(paths, 'env_infos',
                                                       max_len)

    valids = [np.ones_like(path['rewards']) for path in paths]
    valids = np_tensor_utils.pad_tensor_n(valids, max_len)

    lengths = np.stack([path['lengths'] for path in paths])

    return (observations, actions, rewards, dones, returns, valids, lengths,
            env_infos, agent_infos)
