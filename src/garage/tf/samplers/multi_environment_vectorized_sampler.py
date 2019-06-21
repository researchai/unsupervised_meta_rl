"""A sampler that obtains samples from multiple environmennts."""
import itertools
import pickle

from dowel import logger, tabular
import numpy as np

from garage.misc import tensor_utils
from garage.misc.overrides import overrides
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.sampler.utils import truncate_paths
from garage.tf.envs import VecEnvExecutor
from garage.tf.samplers.on_policy_vectorized_sampler import (
    OnPolicyVectorizedSampler)


class MultiEnvironmentVectorizedSampler(OnPolicyVectorizedSampler):
    """This class implements an MultiEnvironmentVectorizedSampler.

    Args:
        algo:
        envs:
        n_envs:
    """

    def __init__(self, algo, envs, n_envs=1):
        super().__init__(algo, env=None, n_envs=n_envs)
        self.envs = envs
        self.n_tasks = len(envs)
        self.env_specs = []
        self.vec_envs = []
        self.obs_augments = np.eye(self.n_tasks)[list(range(self.n_tasks))]

    @overrides
    def start_worker(self):
        """Initialize the sampler."""
        n_envs = self.n_envs
        for env in self.envs:
            envs = [pickle.loads(pickle.dumps(env)) for _ in range(n_envs)]
            vec_env = VecEnvExecutor(
                envs=envs, max_path_length=self.algo.max_path_length)
            env_spec = env.spec
            self.vec_envs.append(vec_env)
            self.env_specs.append(env_spec)

    @overrides
    def shutdown_worker(self):
        """Terminate workers if necessary."""
        for vec_env in self.vec_envs:
            vec_env.close()

    @overrides
    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        """Collect samples for the given iteration number.

        Args:
            itr(int): Iteration number.
            batch_size(int): Number of environment interactions in one batch.
            whole_path(bool): Whether returns the whole paths.

        Returns:
            list: A list of paths from muntiple environments.

        """
        logger.log('Obtaining samples for iteration %d...' % itr)

        if not batch_size:
            batch_size = self.algo.max_path_length * self.n_envs * self.n_tasks
        batch_size_per_task = batch_size // self.n_tasks

        tasks_paths = []

        for i, (env, vec_env, obs_augment) in enumerate(
                zip(self.envs, self.vec_envs, self.obs_augments)):
            paths = []
            n_samples = 0
            obses = vec_env.reset()
            dones = np.asarray([True] * vec_env.num_envs)
            running_paths = [None] * vec_env.num_envs

            pbar = ProgBarCounter(batch_size)
            policy_time = 0
            env_time = 0
            process_time = 0

            policy = self.algo.policy

            import time
            while n_samples < batch_size_per_task:
                t = time.time()
                policy.reset(dones)

                obses_augs = [
                    np.concat([obs, obs_augment], axis=0) for obs in obses
                ]
                actions, agent_infos = policy.get_actions(obses_augs)

                policy_time += time.time() - t
                t = time.time()
                next_obses, rewards, dones, env_infos = vec_env.step(actions)
                env_time += time.time() - t
                t = time.time()

                agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
                env_infos = tensor_utils.split_tensor_dict_list(env_infos)
                if env_infos is None:
                    env_infos = [dict() for _ in range(vec_env.num_envs)]
                if agent_infos is None:
                    agent_infos = [dict() for _ in range(vec_env.num_envs)]
                for idx, observation, action, reward, env_info, agent_info, done in zip(  # noqa: E501
                        itertools.count(), obses_augs, actions, rewards,
                        env_infos, agent_infos, dones):
                    if running_paths[idx] is None:
                        running_paths[idx] = dict(
                            observations=[],
                            actions=[],
                            rewards=[],
                            env_infos=[],
                            agent_infos=[],
                        )
                    running_paths[idx]['observations'].append(observation)
                    running_paths[idx]['actions'].append(action)
                    running_paths[idx]['rewards'].append(reward)
                    running_paths[idx]['env_infos'].append(env_info)
                    running_paths[idx]['agent_infos'].append(agent_info)
                    if done:
                        paths.append(
                            dict(
                                observations=self.env_spec.observation_space.
                                flatten_n(running_paths[idx]['observations']),
                                actions=self.env_spec.action_space.flatten_n(
                                    running_paths[idx]['actions']),
                                rewards=tensor_utils.stack_tensor_list(
                                    running_paths[idx]['rewards']),
                                env_infos=tensor_utils.stack_tensor_dict_list(
                                    running_paths[idx]['env_infos']),
                                agent_infos=tensor_utils.
                                stack_tensor_dict_list(
                                    running_paths[idx]['agent_infos'])))
                        n_samples += len(running_paths[idx]['rewards'])
                        running_paths[idx] = None

                process_time += time.time() - t
                pbar.inc(len(obses))
                obses = next_obses

            tasks_paths.append(paths)

        pbar.stop()

        tabular.record('PolicyExecTime', policy_time)
        tabular.record('EnvExecTime', env_time)
        tabular.record('ProcessExecTime', process_time)

        if whole_paths:
            return tasks_paths
        else:
            return [truncate_paths(paths, batch_size) for paths in tasks_paths]
