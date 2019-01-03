import copy
import numpy as np
import gym

from garage.contrib.exp.core.misc import get_env_spec


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

        self.env_spec = get_env_spec(self._env)
        self.obs_dim = self.env_spec.observation_space.flat_dim
        self.action_dim = self.env_spec.action_space.flat_dim

    def reset(self):
        self.path_idx = [i for i in range(self.n_env)]
        self.observations = []
        self.actions = []
        self.rewards = []
        self.infos = []

        ret_obs = []

        for i in range(self.n_env):
            obs = self.envs[i].reset()
            ret_obs.append(obs)
            self.observations.append(obs.reshape(1, -1))
            self.actions.append(np.zeros((0, self.action_dim)))
            self.rewards.append(np.array([]))
            self.infos.append([])

        return np.array(ret_obs)

    def step(self, actions):
        ret_obs = []

        for i in range(self.n_env):
            idx = self.path_idx[i]
            env = self.envs[i]
            obs, rew, done, info = env.step(actions[i])
            ret_obs.append(obs)

            # print(self.actions[idx], actions[i])
            self.actions[idx] = np.vstack((self.actions[idx], [actions[i]]))
            self.rewards[idx] = np.append(self.rewards[idx], rew)
            self.infos[idx].append(info)
            if not done:
                self.observations[idx] = np.vstack((self.observations[idx], [obs]))
            else:
                idx = idx + 1
                self.path_idx[i] = idx

                self.observations.append(env.reset().reshape(1, -1))
                self.actions.append(np.zeros((0, self.action_dim)))
                self.rewards.append(np.array([]))
                self.infos.append([])

                self._path_count = self._path_count + 1

            self._sample_count = self._sample_count + 1

        return np.array(ret_obs)

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
