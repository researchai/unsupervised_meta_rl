
import itertools
import pickle

import numpy as np

from garage.misc import tensor_utils
import garage.misc.logger as logger
from garage.misc.overrides import overrides
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.tf.samplers import OnPolicyVectorizedSampler


class MultiEnvVectorizedSampler(OnPolicyVectorizedSampler):

    def __init__(self, algo, envs, n_envs=None):
        super().__init__(algo=algo, n_envs=n_envs)
        self.envs = envs
        self.vec_envs = []
    
    @overrides
    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / (self.algo.max_path_length * len(self.envs)))
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            for env in self.envs:
                self.vec_envs.append(self.algo.env.vec_env_executor(
                    n_envs=n_envs, max_path_length=self.algo.max_path_length))
        else:
            for env in self.envs:
                envs = [
                    pickle.loads(pickle.dumps(env))
                    for _ in range(n_envs)
                ]
                self.vec_envs.append(self.vec_env_executor_cls(
                    envs=envs, max_path_length=self.algo.max_path_length))
        self.env_spec = self.envs[0].spec # TODO different environments can have different specs.

    @overrides
    def obtain_samples(self, itr):
        logger.log("Obtaining samples for iteration %d..." % itr)
        all_paths = []

        import time
        pbar = ProgBarCounter(self.algo.batch_size)

        for i, e in enumerate(self.envs):
            vec_env = self.vec_envs[i]

            paths = []
            n_samples = 0
            obses = vec_env.reset()
            dones = np.asarray([True] * self.vec_envs[i].num_envs)
            running_paths = [None] * self.vec_envs[i].num_envs
            
            policy_time = 0
            env_time = 0
            process_time = 0

            policy = self.algo.policy

            batch_size_per_task = self.algo.batch_size // self.algo.policy.n_tasks

            # Rollout here..
            while n_samples < batch_size_per_task:
                t = time.time()
                policy.reset(dones)

                actions, agent_infos = policy.get_actions(obses)

                policy_time += time.time() - t
                t = time.time()
                next_obses, rewards, dones, env_infos = self.vec_envs[i].step(actions)
                env_time += time.time() - t
                t = time.time()

                agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
                env_infos = tensor_utils.split_tensor_dict_list(env_infos)
                if env_infos is None:
                    env_infos = [dict() for _ in range(self.vec_envs[i].num_envs)]
                if agent_infos is None:
                    agent_infos = [dict() for _ in range(self.vec_envs[i].num_envs)]
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
                                agent_infos=tensor_utils.stack_tensor_dict_list(
                                    running_paths[idx]["agent_infos"])))
                        n_samples += len(running_paths[idx]["rewards"])
                        running_paths[idx] = None

                process_time += time.time() - t
                pbar.inc(len(obses))
                obses = next_obses

            all_paths.append(paths)

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        return all_paths

    @overrides
    def process_samples(self, itr, paths):
        all_samples = []
        for p in paths:
            processed = super().process_samples(itr, p)
            all_samples.append(processed)
        return all_samples
