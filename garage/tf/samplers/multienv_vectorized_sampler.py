"""On policy sampler for a list of environments."""
import itertools
import pickle

import numpy as np

from garage.misc import tensor_utils
from garage.logger import logger, tabular
from garage.misc.overrides import overrides
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.tf.envs import VecEnvExecutor
from garage.tf.samplers.on_policy_vectorized_sampler import (
    OnPolicyVectorizedSampler)


class MultiEnvVectorizedSampler(OnPolicyVectorizedSampler):
    """
    Multi-Environment Vectorized Sampler.

    This sampler is just a multi-envrionment version
    of OnPolicyVectorizedSampler. It takes a list of
    different environment and sample from them in the
    same way as an OnPolicyVectorizedSampler. This is
    used for meta RL algorithms which need to sample
    from a set of MDP's.

    Args:
        algo: An meta RL algorithm.
        envs: A list of environments.
        n_envs: The number of environments to be created
            for each VecEnvExecutor.
    """

    def __init__(self, algo, envs, n_envs=2):
        super().__init__(algo=algo, n_envs=n_envs, env=envs[0])
        self.envs = envs
        self.vec_envs = []
        self.meta_batch_size = self.algo.policy.meta_batch_size
        self._random_indices = None

    @overrides
    def start_worker(self):
        """Create a list of vectorized executors."""
        n_envs = self.n_envs

        for env in self.envs:
            envs = [pickle.loads(pickle.dumps(env)) for _ in range(n_envs)]
            self.vec_envs.append(
                VecEnvExecutor(
                    envs=envs, max_path_length=self.algo.max_path_length))
        self.env_spec = self.envs[0].spec

    @overrides
    def obtain_samples(self, itr, batch_size=None, adaptation_data=None):
        """
        Sample from environments.

        Args:
            itr: Iteration number.
            batch_size: The number of samples.
        """
        logger.log("Obtaining samples for iteration %d..." % itr)
        all_paths = []

        if not batch_size:
            batch_size = self.algo.max_path_length * self.n_envs * len(
                self.envs)
        batch_size_per_task = self.algo.max_path_length * self.n_envs

        import time
        pbar = ProgBarCounter(batch_size)
        if adaptation_data is None:
            self._random_indices = np.random.randint(low=0, high=len(self.envs), size=self.meta_batch_size)

        for i in range(self.meta_batch_size):
            vec_env = self.vec_envs[self._random_indices[i]]

            paths = []
            n_samples = 0
            obses = vec_env.reset()
            dones = np.asarray([True] * vec_env.num_envs)
            running_paths = [None] * vec_env.num_envs

            policy_time = 0
            env_time = 0
            process_time = 0

            policy = self.algo.policy

            while n_samples < batch_size_per_task:
                t = time.time()
                policy.reset(dones)

                if adaptation_data is None:
                    actions, agent_infos = policy.get_actions(obses)
                else:
                    actions, agent_infos = policy.get_actions_with_adaptation_data(obses, adaptation_data[i])

                policy_time += time.time() - t
                t = time.time()
                next_obses, rewards, dones, env_infos = vec_env.step(
                    actions)
                env_time += time.time() - t
                t = time.time()

                agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
                env_infos = tensor_utils.split_tensor_dict_list(env_infos)
                if env_infos is None:
                    env_infos = [
                        dict() for _ in range(vec_env.num_envs)
                    ]
                if agent_infos is None:
                    agent_infos = [
                        dict() for _ in range(vec_env.num_envs)
                    ]
                for idx, observation, action, reward, env_info, agent_info, done in zip(  # noqa: E501
                        itertools.count(), obses, actions, rewards, env_infos,
                        agent_infos, dones):
                    if running_paths[idx] is None:
                        running_paths[idx] = dict(
                            observations=[],
                            actions=[],
                            rewards=[],
                            env_infos=[],
                            agent_infos=[],
                        )
                    running_paths[idx]["observations"].append(observation)
                    running_paths[idx]["actions"].append(action)
                    running_paths[idx]["rewards"].append(reward)
                    running_paths[idx]["env_infos"].append(env_info)
                    running_paths[idx]["agent_infos"].append(agent_info)
                    if done:
                        paths.append(
                            dict(
                                observations=self.env_spec.observation_space.
                                flatten_n(running_paths[idx]["observations"]),
                                actions=self.env_spec.action_space.flatten_n(
                                    running_paths[idx]["actions"]),
                                rewards=tensor_utils.stack_tensor_list(
                                    running_paths[idx]["rewards"]),
                                env_infos=tensor_utils.stack_tensor_dict_list(
                                    running_paths[idx]["env_infos"]),
                                agent_infos=tensor_utils.
                                stack_tensor_dict_list(
                                    running_paths[idx]["agent_infos"])))
                        n_samples += len(running_paths[idx]["rewards"])
                        running_paths[idx] = None

                process_time += time.time() - t
                pbar.inc(len(obses))
                obses = next_obses

            all_paths.append(paths)

        pbar.stop()

        tabular.record("PolicyExecTime", policy_time)
        tabular.record("EnvExecTime", env_time)
        tabular.record("ProcessExecTime", process_time)

        return all_paths

    @overrides
    def process_samples(self, itr, paths):
        """Process samples."""
        # This is super hacky...
        processed = super().process_samples(itr, paths)
        # fit the baseline
        self.algo.fit_baseline_once(processed)
        # recalculate baseline value...
        all_path_baselines = [
            self.algo.baseline.predict(path) for path in paths
        ]

        baselines = tensor_utils.pad_tensor_n(all_path_baselines, self.algo.max_path_length)
        processed['baselines'] = baselines
        return processed


    @overrides
    def shutdown_worker(self):
        for vec_env in self.vec_envs:
            vec_env.close()
