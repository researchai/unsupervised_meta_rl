import akro
import gym
import numpy as np
from smac.env import StarCraft2Env

from garage.envs import GarageEnv


class SC2EnvWrapper(GarageEnv):
    """
    Wraps StarCraft2Env inside GarageEnv
    TODO: normalization - more precision on obs, reward, action ranges
    """

    def __init__(self,
                 map_name="8m",
                 step_mul=8,
                 move_amount=2,
                 difficulty="7",
                 game_version=None,
                 seed=None,
                 continuing_episode=False,
                 obs_all_health=True,
                 obs_own_health=True,
                 obs_last_action=False,
                 obs_pathing_grid=False,
                 obs_terrain_height=False,
                 obs_instead_of_state=False,
                 obs_timestep_number=False,
                 state_last_action=True,
                 state_timestep_number=False,
                 reward_sparse=False,
                 reward_only_positive=True,
                 reward_death_value=10,
                 reward_win=200,
                 reward_defeat=0,
                 reward_negative_scale=0.5,
                 reward_scale=True,
                 reward_scale_rate=20,
                 replay_dir="",
                 replay_prefix="",
                 window_size_x=1920,
                 window_size_y=1200,
                 heuristic_ai=False,
                 heuristic_rest=False,
                 debug=False,
                 ):
        self.sc2_env = StarCraft2Env(map_name, step_mul, move_amount,
                                     difficulty, game_version, seed,
                                     continuing_episode, obs_all_health,
                                     obs_own_health, obs_last_action,
                                     obs_pathing_grid, obs_terrain_height,
                                     obs_instead_of_state, obs_timestep_number,
                                     state_last_action, state_timestep_number,
                                     reward_sparse, reward_only_positive,
                                     reward_death_value, reward_win,
                                     reward_defeat, reward_negative_scale,
                                     reward_scale, reward_scale_rate,
                                     replay_dir, replay_prefix, window_size_x,
                                     window_size_y, heuristic_ai,
                                     heuristic_rest, debug)
        self.env_info = self.sc2_env.get_env_info()

        self.n_agents = self.env_info["n_agents"]
        self.n_actions = self.env_info["n_actions"]
        self.obs_shape = self.env_info["obs_shape"]
        self.concat_obs_shape = self.obs_shape * self.n_agents
            # concatenate observation for the convenience of implementation
        self.concat_action_shape = self.n_actions * self.n_agents
        self.state_shape = self.env_info["state_shape"]
        self.episode_limit = self.env_info["episode_limit"]

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @property
    def spec(self):
        print("attempted to get environment specifications from StarCraft2 -"
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

    def _get_action_space(self):
        return akro.from_gym(gym.spaces.Box(0, self.n_actions,
                                            self.concat_action_shape))

    def _get_observation_space(self):
        return akro.from_gym(gym.spaces.Box(-np.inf, np.inf,
                                            self.concat_obs_shape))


sc2_env_wrap = SC2EnvWrapper
