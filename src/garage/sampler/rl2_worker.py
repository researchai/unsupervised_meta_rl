"""Worker class used in all Samplers."""
from collections import defaultdict

import gym
import numpy as np

from garage import TrajectoryBatch
from garage.sampler.worker import DefaultWorker


class RL2Worker(DefaultWorker):
    """Initialize a worker.

    Args:
        seed(int): The seed to use to intialize random number generators.
        max_path_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(gym.Env or None): The worker's environment.

    """
    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_path_length,
            worker_number):
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)

    def collect_rollout(self):
        """Collect the current rollout, clearing the internal buffer.

        Returns:
            garage.TrajectoryBatch: A batch of the trajectories completed since
                the last call to collect_rollout().

        """
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
        agent_infos['batch_idx'] = np.full(len(observations), self._worker_number)
        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)
        lengths = self._lengths
        self._lengths = []
        return TrajectoryBatch(self.env.spec, np.asarray(observations),
                               np.asarray(last_observations),
                               np.asarray(actions), np.asarray(rewards),
                               np.asarray(terminals), dict(env_infos),
                               dict(agent_infos), np.asarray(lengths,
                                                             dtype='i'))