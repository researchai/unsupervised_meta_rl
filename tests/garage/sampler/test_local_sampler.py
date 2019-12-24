from unittest.mock import Mock

import numpy as np

from garage.envs.grid_world_env import GridWorldEnv
from garage.np.policies import ScriptedPolicy
from garage.sampler import LocalSampler, OnPolicyVectorizedSampler, \
    WorkerFactory
from garage.tf.envs import TfEnv


class TestSampler:
    """
    Uses mock policy for 4x4 gridworldenv
    '4x4': [
        'SFFF',
        'FHFH',
        'FFFH',
        'HFFG'
    ]
    0: left
    1: down
    2: right
    3: up
    -1: no move
    'S' : starting point
    'F' or '.': free space
    'W' or 'x': wall
    'H' or 'o': hole (terminates episode)
    'G' : goal
    [2,2,1,0,3,1,1,1,2,2,1,1,1,2,2,1]
    """

    def setup_method(self):
        self.env = TfEnv(GridWorldEnv(desc='4x4'))
        self.policy = ScriptedPolicy(
            scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])
        self.algo = Mock(env_spec=self.env.spec,
                         policy=self.policy,
                         max_path_length=16)

    def teardown_method(self):
        self.env.close()

    def test_ray_batch_sampler(self):
        workers = WorkerFactory(seed=100,
                                max_path_length=self.algo.max_path_length)
        sampler1 = LocalSampler.from_worker_factory(workers, self.policy,
                                                    self.env)
        sampler2 = OnPolicyVectorizedSampler(self.algo, self.env)
        sampler2.start_worker()
        trajs1 = sampler1.obtain_samples(
            0, 1000, tuple(self.algo.policy.get_param_values()))
        trajs2 = sampler2.obtain_samples(0, 1000)
        # pylint: disable=superfluous-parens
        assert trajs1.observations.shape[0] >= 1000
        assert trajs1.actions.shape[0] >= 1000
        assert (sum(trajs1.rewards[:trajs1.lengths[0]]) == sum(
            trajs2[0]['rewards']) == 1)

        true_obs = np.array([0, 1, 2, 6, 10, 14])
        true_actions = np.array([2, 2, 1, 1, 1, 2])
        true_rewards = np.array([0, 0, 0, 0, 0, 1])
        start = 0
        for length in trajs1.lengths:
            observations = trajs1.observations[start:start + length]
            actions = trajs1.actions[start:start + length]
            rewards = trajs1.rewards[start:start + length]
            assert np.array_equal(observations, true_obs)
            assert np.array_equal(actions, true_actions)
            assert np.array_equal(rewards, true_rewards)
            start += length
        sampler1.shutdown_worker()
        sampler2.shutdown_worker()