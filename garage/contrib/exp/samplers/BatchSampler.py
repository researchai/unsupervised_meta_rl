import copy
import numpy as np
import gym


class BatchSampler:
    def __init__(self, env: gym.Env, n_env=1, max_path_length=100):
        self._env = env
        self.n_env = n_env
        self.max_path_length = max_path_length

        self.envs = [copy.deepcopy(env) for _ in range(n_env)]
        self.path_idx = [i for i in range(self.n_env)]
        self.observations = []
        self.actions = []
        self.rewards = []
        self.infos = []

        self._sample_count = 0
        self._path_count = 0

    def reset(self):
        self.path_idx = [i for i in range(self.n_env)]
        self.observations = []
        self.actions = []
        self.rewards = []
        self.infos = []

        for i in range(self.n_env):
            self.observations.append([self.envs[i].reset()])
            self.actions.append([])
            self.rewards.append([])
            self.infos.append([])

    def step(self, actions):
        for i in range(self.n_env):
            idx = self.path_idx[i]
            env = self.envs[i]
            obs, rew, done, info = env.step(actions[i])

            self.rewards[idx].append(obs)
            self.infos[idx].append(info)
            if not done:
                self.observations[idx].append(obs)
            else:
                idx = idx + 1
                self.path_idx[i] = idx

                self.observations.append([env.reset()])
                self.actions.append([])
                self.rewards.append([])
                self.infos.append([])

                self._path_count = self._path_count + 1

            self._sample_count = self._sample_count + 1

    def get_samples(self):
        return {
            'observations': self.observations.copy(),
            'actions': self.actions.copy(),
            'rewards': self.rewards.copy(),
            'infos': self.infos.copy()
        }

    @property
    def sample_count(self):
        return self._sample_count

    @property
    def path_count(self):
        return self._path_count
