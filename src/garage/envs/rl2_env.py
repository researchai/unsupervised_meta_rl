"""Wrapper class that converts gym.Env into GarageEnv."""

import collections
import copy
import numpy as np

import akro
import gym
from garage.envs.env_spec import EnvSpec


class RL2Env(gym.Wrapper):
    """Returns an abstract Garage wrapper class for gym.Env.

    In order to provide pickling (serialization) and parameterization
    for gym.Envs, they must be wrapped with a GarageEnv. This ensures
    compatibility with existing samplers and checkpointing when the
    envs are passed internally around garage.

    Furthermore, classes inheriting from GarageEnv should silently
    convert action_space and observation_space from gym.Spaces to
    akro.spaces.

    Args:
        env (gym.Env): An env that will be wrapped
        env_name (str): If the env_name is speficied, a gym environment
            with that name will be created. If such an environment does not
            exist, a `gym.error` is thrown.

    """

    def __init__(self, env, max_obs_dim=None):
        super().__init__(env)
        self.max_obs_dim = max_obs_dim
        action_space = akro.from_gym(self.env.action_space)
        observation_space = akro.from_gym(self._create_rl2_obs_space(env))
        self.spec = EnvSpec(action_space=action_space,
                            observation_space=observation_space)

    def _create_rl2_obs_space(self, env):
        obs_flat_dim = np.prod(env.observation_space.shape)
        action_flat_dim = np.prod(env.action_space.shape)
        if self.max_obs_dim is not None:
            obs_flat_dim = self.max_obs_dim
        return gym.spaces.Box(low=-np.inf,
                              high=np.inf,
                              shape=(obs_flat_dim + action_flat_dim + 1 + 1,))

    def reset(self):
        obs = self.env.reset()
        # pad zeros if needed for running ML45
        if self.max_obs_dim is not None:
            obs = np.concatenate([obs, np.zeros(self.max_obs_dim - obs.shape[0])])
        return np.concatenate([obs, np.zeros(self.env.action_space.shape), [0], [0]])

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if self.max_obs_dim is not None:
            next_obs = np.concatenate([next_obs, np.zeros(self.max_obs_dim - next_obs.shape[0])])
        next_obs = np.concatenate([next_obs, action, [reward], [done]])
        return next_obs, reward, done, info

    @property
    def num_tasks(self):
        return self.env.num_tasks

    @property
    def _task_names(self):
        return self.env._task_names
