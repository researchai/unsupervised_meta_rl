import itertools
import pickle

import numpy as np

from garage.misc import tensor_utils
import garage.misc.logger as logger
from garage.misc.overrides import overrides
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.tf.envs import VecEnvExecutor
from garage.tf.samplers import BatchSampler


class OnPolicyVectorizedSampler(BatchSampler):
    def __init__(self, env, n_envs=None):
        # super(OnPolicyVectorizedSampler, self).__init__(algo)
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(
                n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [
                pickle.loads(pickle.dumps(self.algo.env))
                for _ in range(n_envs)
            ]
            self.vec_env = VecEnvExecutor(
                envs=envs, max_path_length=self.algo.max_path_length)
        self.env_spec = self.algo.env.spec

        self.obs_dim = self.env_spec.observation_space.flat_dim
        self.action_dim = self.env_spec.actions_space.flat_dim

        self.paths = []
        self.running_paths = [([], [], [], []) for i in range(n_envs)]
        self.sample_count = 0
        self.path_count = 0
        self.last_obses = self.vec_env.reset()

    def reset(self):
        self.paths = []
        self.running_paths = [([], [], [], []) for i in range(n_envs)]
        self.sample_count = 0
        self.path_count = 0
        self.last_obses = self.vec_env.reset()

    @overrides
    def start_worker(self):
        pass

    @overrides
    def shutdown_worker(self):
        self.vec_env.close()

    def step(self, actions):
        ret_obses = []
        obses, rewards, dones, infos = self.vec_env.step(actions)

        infos = tensor_utils.split_tensor_dict_list(infos)
        if infos is None:
            infos = [dict() for _ in range(self.n_envs)]

        for i, obs, action, reward, info, done in zip(  # noqa: E501
                itertools.count(), self.last_obses, actions, rewards, infos, dones):
            
            obs_p, action_p, reward_p, info_p = self.running_paths[i]
            obs_p.append(obs)
            action_p.append(action)
            reward_p.append(reward)
            info_p.append(info)

            if done:
                self.paths.append({
                    "observations": self.env_spec.observation_space.flatten_n(obs_p),
                    "actions": self.env_spec.action_space.flatten_n(action_p),
                    "rewards": tensor_utils.stack_tensor_list(reward_p),
                    "env_infos": tensor_utils.stack_tensor_dict_list(info_p)
                })
                self.sampler_count += len(reward_p)
                self.path_count += 1
                self.running_paths[i] = [(), (), (), ()]
                obses[i] = self.vec_env.reset(i)

        self.last_obses = obses

        return obses
    
    def get_samples(self):
        return self.paths

    @property
    def sample_count(self):
        return self._sampler_count

    @overrides
    def obtain_samples(self):
        paths = []
        n_samples = 0
        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy

        import time
        while n_samples < self.algo.batch_size:
            t = time.time()
            policy.reset(dones)

            actions, agent_infos = policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t
            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for i, observation, action, reward, env_info, agent_info, done in zip(  # noqa: E501
                    itertools.count(), obses, actions, rewards, env_infos,
                    agent_infos, dones):
                if running_paths[i] is None:
                    running_paths[i] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[i]["observations"].append(observation)
                running_paths[i]["actions"].append(action)
                running_paths[i]["rewards"].append(reward)
                running_paths[i]["env_infos"].append(env_info)
                running_paths[i]["agent_infos"].append(agent_info)
                if done:
                    paths.append(
                        dict(
                            observations=self.env_spec.observation_space.
                            flatten_n(running_paths[i]["observations"]),
                            actions=self.env_spec.action_space.flatten_n(
                                running_paths[i]["actions"]),
                            rewards=tensor_utils.stack_tensor_list(
                                running_paths[i]["rewards"]),
                            env_infos=tensor_utils.stack_tensor_dict_list(
                                running_paths[i]["env_infos"]),
                            agent_infos=tensor_utils.stack_tensor_dict_list(
                                running_paths[i]["agent_infos"])))
                    n_samples += len(running_paths[i]["rewards"])
                    running_paths[i] = None

            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        return paths