#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm."""
import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper  # pylint: disable=E0401

from garage.envs import normalize
from garage.envs.wrappers import Grayscale
from garage.envs.wrappers import Resize
from garage.envs.wrappers import StackFrames
from garage.experiment import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianConvBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalConvPolicyWithModel


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = gym.make('Acrobot-v1')
        env.reset()
        env = PixelObservationWrapper(env, pixels_only=True)
        size = env.observation_space.spaces['pixels'].shape
        env.observation_space = gym.spaces.Box(0,
                                               255,
                                               shape=size,
                                               dtype='float32')
        env = Grayscale(env)
        env = Resize(env, 84, 84)
        env = StackFrames(env, 2)
        env = TfEnv(normalize(env))

        policy = CategoricalConvPolicyWithModel(env_spec=env.spec,
                                                conv_filters=(16, 16),
                                                conv_filter_sizes=(3, 5),
                                                conv_strides=(2, 2),
                                                conv_pad='VALID',
                                                hidden_sizes=(16, 16))

        baseline = GaussianConvBaseline(env_spec=env.spec,
                                        regressor_args=dict(
                                            conv_filters=(32, 64),
                                            conv_filter_sizes=(8, 4),
                                            conv_strides=(4, 2),
                                            conv_pads=('VALID', 'VALID'),
                                            hidden_sizes=(32, 32),
                                            use_trust_region=True))

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=100,
                   discount=0.99,
                   max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
