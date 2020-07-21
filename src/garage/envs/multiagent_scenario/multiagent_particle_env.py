import akro
import numpy as np
from gym.spaces import Box, Tuple, Discrete

import garage.envs.multiagent_scenario.scenarios as scenarios
from garage.envs.multiagent_scenario.environment import MultiAgentEnv
from garage.envs import GarageEnv
from garage.envs.multiagent_scenario.multi_discrete import MultiDiscrete


class MultiParticleEnvWrapper(GarageEnv):
    """
     Wraps multiagent.environment.MultiAgent Envinside GarageEnv
     TODO: current action and observation not arko - needs to be decided in the
           future rollout implementation
    """

    def __init__(self,
                 scenario_name,
                 benchmark=False):
        self.env = self._make_env(scenario_name, benchmark)
        self.n_agents = self.env.n
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = MultiAgentEnv.metadata

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
        return self.env.step(actions)

    def reset(self, **kwargs):
        return self.env.reset()

    def render(self, mode='human', **kwargs):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

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
        scenario = scenarios.load(scenario_name + "py").Scenario()
        world = scenario.make_world()

        if benchmark:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                                scenario.observation, scenario.benchmark_data)
        else:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                                scenario.observation)
        return env

    # def _get_action_space(self):
    #
    #     # configure spaces
    #     self.action_space = []
    #     self.observation_space = []
    #     for agent in self.env.agents:
    #         total_action_space = []
    #         # physical action space
    #         if self.discrete_action_space:
    #             u_action_space = Discrete(self.env.world.dim_p * 2 + 1)
    #         else:
    #             u_action_space = Box(low=-agent.u_range,
    #                                  high=+agent.u_range,
    #                                  shape=(self.env.world.dim_p,),
    #                                  dtype=np.float32)
    #         if agent.movable:
    #             total_action_space.append(u_action_space)
    #         # communication action space
    #         if self.discrete_action_space:
    #             c_action_space = Discrete(self.env.world.dim_c)
    #         else:
    #             c_action_space = Box(low=0.0, high=1.0,
    #                                  shape=(self.env.world.dim_c,),
    #                                  dtype=np.float32)
    #         if not agent.silent:
    #             total_action_space.append(c_action_space)
    #         # total action space
    #         if len(total_action_space) > 1:
    #             # all action spaces are discrete, so simplify to MultiDiscrete action space
    #             if all([isinstance(act_space, Discrete) for act_space in
    #                     total_action_space]):
    #                 act_space = MultiDiscrete([[0, act_space.n - 1]
    #                                            for act_space in total_action_space])
    #             else:
    #                 act_space = Tuple(total_action_space)
    #             self.action_space.append(act_space)
    #         else:
    #             self.action_space.append(total_action_space[0])
    #         # observation space
    #         obs_dim = len(self.env.observation_callback(agent, self.world))
    #         self.observation_space.append(Box(low=-np.inf,
    #                                           high=+np.inf,
    #                                           shape=(obs_dim,),
    #                                           dtype=np.float32))
    #         agent.action.c = np.zeros(self.world.dim_c)
    #
    #     return akro.from_gym(Tuple([Discrete(self.n_actions)
    #                                 for _ in range(self.n_agents)]))
    #
    # def _get_observation_space(self):
    #     return akro.from_gym(Box(-np.inf, np.inf, self.concat_obs_shape))


multi_particle_env_wrap = MultiParticleEnvWrapper
