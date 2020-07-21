import akro
import numpy as np
from gym.spaces import Box, Tuple, Discrete
import multiagent

from garage.envs import GarageEnv


class MultiParticleEnvWrapper(GarageEnv):
    """
     Wraps multiagent.environment.MultiAgent Envinside GarageEnv
    """

    def __init__(self,
                 scenario_name):

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError( "attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @property
    def spec(self):
        print(
            "attempted to get environment specifications from StarCraft2 -"
            "currently unavailable, returned None")
        return None

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        return self.sc2_env.step(actions)

    def reset(self, **kwargs):
        return self.sc2_env.reset()

    def render(self, mode='human', **kwargs):
        raise NotImplementedError

    def close(self):
        return self.sc2_env.close()

    def seed(self, seed=None):
        return self.sc2_env.seed()

    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError
        # return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        print("attempted to get unwrapped var from StarCraft2 -"
              "currently unavailable, returned None")
        return None

    def _make_env(self, scenario_name, benchmark=False):  # adapted from make_env.py from multiagent repo
        from multiagent.en

    def _get_action_space(self):
        return akro.from_gym(Tuple([Discrete(self.n_actions)
                                    for _ in range(self.n_agents)]))

    def _get_observation_space(self):
        return akro.from_gym(Box(-np.inf, np.inf, self.concat_obs_shape))


multi_particle_env_wrap = MultiParticleEnvWrapper
