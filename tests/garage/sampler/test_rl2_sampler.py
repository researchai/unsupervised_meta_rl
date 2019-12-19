import gym
import numpy as np
import pytest

from garage.envs import HalfCheetahVelEnv
from garage.envs import RL2Env
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler.rl2_sampler import RL2Sampler
from garage.tf.algos import PPO
from garage.tf.algos import RL2
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianGRUPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


configs = [(1, None, 4, 4), (3, None, 12, 12), (2, 10, 5, 10)]


class TestRL2Sampler(TfGraphTestCase):

    @pytest.mark.parametrize('cpus, n_envs, meta_batch_size, expected_n_envs', [*configs])
    def test_rl2_sampler_n_envs(self, cpus, n_envs, meta_batch_size,
                                                 expected_n_envs):
        with LocalTFRunner(snapshot_config, sess=self.sess,
                           max_cpus=cpus) as runner:
            env = RL2Env(env=HalfCheetahVelEnv())

            policy = GaussianGRUPolicy(env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            inner_algo = PPO(env_spec=env.spec,
                             policy=policy,
                             baseline=baseline,
                             max_path_length=100,
                             discount=0.99,
                             lr_clip_range=0.2,
                             optimizer_args=dict(max_epochs=5))

            algo = RL2(policy=policy, inner_algo=inner_algo, max_path_length=100)

            runner.setup(algo, env, sampler_cls=RL2Sampler, sampler_args=dict(
                meta_batch_size=meta_batch_size, episode_per_task=1, n_envs=n_envs))

            assert isinstance(runner._sampler, RL2Sampler)
            assert runner._sampler._n_envs == expected_n_envs

            env.close()

    @pytest.mark.parametrize('n_envs, meta_batch_size', [(1, 1), (2, 1), (10, 5)])
    def test_rl2_sampler_tasks(self, n_envs, meta_batch_size):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = RL2Env(env=HalfCheetahVelEnv())

            policy = GaussianGRUPolicy(env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            inner_algo = PPO(env_spec=env.spec,
                             policy=policy,
                             baseline=baseline,
                             max_path_length=100,
                             discount=0.99,
                             lr_clip_range=0.2,
                             optimizer_args=dict(max_epochs=5))

            algo = RL2(policy=policy, inner_algo=inner_algo, max_path_length=100)

            runner.setup(algo, env, sampler_cls=RL2Sampler, sampler_args=dict(
                meta_batch_size=meta_batch_size, episode_per_task=1, n_envs=n_envs))
            runner.train(n_epochs=1, batch_size=None)

            assert isinstance(runner._sampler, RL2Sampler)
            assert len(runner._sampler._vec_env.envs) == n_envs
            env_tasks = [env.get_task() for env in runner._sampler._vec_env.envs]
            assert len(np.unique(env_tasks)) == meta_batch_size
            env.close()
