"""Skill Worker class."""
from collections import defaultdict

import numpy as np

from garage import SkillTrajectoryBatch
from garage.sampler import DefaultWorker


class SkillWorker(DefaultWorker):
    def __init__(
            self,
            *,
            seed,
            skills_num,
            max_path_length,
            worker_number):
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)

        self._prob_skill = np.full(skills_num, 1.0 / skills_num)
        self._skills_num = skills_num
        self._skills = []
        self._states = []
        self._last_states = []
        self._cur_z = None
        self._prev_s = None

    # def worker_init(self):

    # def update_agent(self, agent_update):

    # def update_env(self, env_update):

    def start_rollout(self):
        self._path_length = 0
        self._prev_s = self.env.reset()
        self._cur_z = self._sample_skill()
        self._prev_obs = np.concatenate((self._prev_s, self._cur_z), dim=1)
        self.agent.reset()

    def step_rollout(self):
        if self._path_length < self._max_path_length:
            a, agent_info = self.agent.get_action(self._prev_obs)
            next_s, r, d, env_info = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._rewards.append(r)
            self._actions.append(a)
            self._skills.append(self._cur_z)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k, v in env_info.items():
                self._env_infos[k].append(v)
            self._path_length += 1
            self._terminals.append(d)
            if not d:
                self._prev_s = next_s
                self._prev_obs = np.concatenate((next_s, self._cur_z), dim=1)
                return False
        self._lengths.append(self._path_length)
        self._last_states.append(self._prev_s)
        self._last_observations.append(self._prev_obs)

    def collect_rollout(self):
        states = self._states
        self._states = []
        last_states = self._last_states
        self._last_states = []
        skills = self._skills
        self._skills = []
        observations = self._observations
        self._observations = []
        last_observations = self._last_observations
        self._last_observations = []
        actions = self._actions
        self._actions = []
        rewards = self._rewards
        self._rewards = []
        terminals = self._terminals
        self._terminals = []
        env_infos = self._env_infos
        self._env_infos = defaultdict(list)
        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)
        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)
        lengths = self._lengths
        self._lengths = []

        return SkillTrajectoryBatch(self.env.spec, self._skills_num,
                                    np.asarray(skills), np.asarray(states),
                                    np.asarray(last_states), np.asarray(observations),
                                    np.asarray(actions), np.asarray(rewards),
                                    np.asarray(terminals), dict(env_infos),
                                    dict(agent_infos), np.asarray(lengths, dtype='i'))

    # def rollout(self)

    # def shutdown(self)

    def _sample_skill(self):  # uniform dist. in order to maximize entropy
        return np.random.choice(self._skills_num, self._prob_skill)
