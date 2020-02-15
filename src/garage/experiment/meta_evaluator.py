"""Evaluator which tests Meta-RL algorithms on test environments."""
from dowel import logger, tabular
import numpy as np

from garage import log_multitask_performance, TrajectoryBatch
from garage.sampler import LocalSampler
from garage.sampler import RaySampler
from garage.sampler.rl2_worker import RL2Worker


class MetaEvaluator:
    """Evaluates Meta-RL algorithms on test environments.

    Args:
        runner (garage.experiment.LocalRunner): A runner capable of running
            policies from the (meta) algorithm. Can be the same runner used by
            the algorithm. Does not use runner.obtain_samples, and so does not
            affect TotalEnvSteps.
        test_task_sampler (garage.experiment.TaskSampler): Sampler for test
            tasks. To demonstrate the effectiveness of a meta-learning method,
            these should be different from the training tasks.
        max_path_length (int): Maximum path length used for evaluation
            trajectories.
        n_test_tasks (int or None): Number of test tasks to sample each time
            evaluation is performed. Note that tasks are sampled "without
            replacement". If None, is set to `test_task_sampler.n_tasks`.
        n_exploration_traj (int): Number of trajectories to gather from the
            exploration policy before requesting the meta algorithm to produce
            an adapted policy.
        prefix (str): Prefix to use when logging. Defaults to MetaTest. For
            example, this results in logging the key 'MetaTest/SuccessRate'.
            If not set to `MetaTest`, it should probably be set to `MetaTrain`.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 runner,
                 *,
                 sampler_cls=LocalSampler,
                 test_task_sampler,
                 max_path_length,
                 n_test_tasks=None,
                 rollout_per_task=10,
                 n_exploration_traj=10,
                 n_test_rollouts=10,
                 prefix='MetaTest',
                 task_name_map={}):
        self._test_task_sampler = test_task_sampler
        if n_test_tasks is None:
            n_test_tasks = 10 * test_task_sampler.n_tasks
        self._n_test_tasks = n_test_tasks
        self._n_exploration_traj = n_exploration_traj
        self._n_test_rollouts = n_test_rollouts
        self._rollout_per_task = rollout_per_task
        self._max_path_length = max_path_length
        self._test_sampler = runner.make_sampler(
            sampler_cls=sampler_cls,
            n_workers=n_test_tasks,
            max_path_length=max_path_length,
            worker_class=RL2Worker,
            policy=runner._algo.get_exploration_policy(),
            env=self._test_task_sampler._env,
            sampler_args=dict(
                n_paths_per_trial=rollout_per_task)
            )
        self._eval_itr = 0
        self._prefix = prefix
        self._task_name_map = task_name_map

    def evaluate(self, algo):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (garage.np.algos.MetaRLAlgorithm): The algorithm to evaluate.

        """
        adapted_trajectories = []

        logger.log('Sampling for adapation and meta-testing...')

        for env_up in self._test_task_sampler.sample(self._n_test_tasks):
            policy = algo.get_exploration_policy()
            traj = TrajectoryBatch.concatenate(*[
                self._test_sampler.obtain_samples(self._eval_itr, 1, policy,
                                                  env_up)
                for _ in range(self._n_exploration_traj)
            ])
            adapted_policy = algo.adapt_policy(policy, traj)
            adapted_hidden_state = adapted_policy._prev_hiddens[:]

            for _ in range(self._n_test_rollouts):
                policy._policy._prev_hiddens = adapted_hidden_state[:]
                adapted_traj = self._test_sampler.obtain_samples(
                    self._eval_itr,
                    1,
                    adapted_policy.get_param_values())
                adapted_trajectories.append(adapted_traj)

        logger.log('Finished meta-testing...')

        with tabular.prefix(self._prefix + '/' if self._prefix else ''):
            log_multitask_performance(self._eval_itr,
                                      TrajectoryBatch.concatenate(
                                          *adapted_trajectories),
                                      getattr(algo, 'discount', 1.0),
                                      self._task_name_map)
        self._eval_itr += 1
