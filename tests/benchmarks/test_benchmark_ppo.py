'''
This script creates a regression test over garage-PPO and baselines-PPO.

Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length.
'''
import datetime
import multiprocessing
import os.path as osp
import random

import dowel
from dowel import logger as dowel_logger
import gym
import pytest
import tensorflow as tf

from garage.envs import normalize
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames
from garage.experiment import deterministic
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import CategoricalLSTMPolicy
from garage.tf.policies import CategoricalLSTMPolicyWithModel
from tests.fixtures import snapshot_config
import tests.helpers as Rh
from tests.wrappers import AutoStopEnv


num_of_trials = 1

params = {
    'n_epochs': 3,
    'max_path_length': 1000
}

class TestBenchmarkPPO:
    '''Compare benchmarks between garage and baselines.'''

    @pytest.mark.huge
    def test_benchmark_ppo(self):
        '''
        Compare benchmarks between garage and baselines.

        :return:
        '''
        envs = ['Breakout-ramNoFrameskip-v4']
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/ppo/%s/' % timestamp
        result_json = {}
        for env_id in envs:
            env = gym.make(env_id)
            seeds = random.sample(range(100), num_of_trials)

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))
            relplt_file = osp.join(benchmark_dir,
                                   '{}_benchmark_mean.png'.format(env_id))
            garage_csvs = []
            garage_with_model_csvs = []

            for trial in range(num_of_trials):
                seed = seeds[trial]

                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                garage_dir = trial_dir + '/garage'
                garage_with_model_dir = trial_dir + '/garage_with_model'

                env = Noop(env, noop_max=30)
                env = MaxAndSkip(env, skip=4)
                env = EpisodicLife(env)
                if 'FIRE' in env.unwrapped.get_action_meanings():
                    env = FireReset(env)
                env = ClipReward(env)

                with tf.Graph().as_default():
                    # Run baselines algorithms
                    env.reset()
                    garage_with_model_csv = run_garage_with_model(env, seed, garage_with_model_dir)

                with tf.Graph().as_default():
                    # Run garage algorithms
                    env.reset()
                    garage_csv = run_garage(env, seed, garage_dir)

                garage_csvs.append(garage_csv)
                garage_with_model_csvs.append(garage_with_model_csv)

            env.close()

            Rh.plot(g_csvs=garage_csvs,
                    b_csvs=garage_with_model_csvs,
                    g_x='Iteration',
                    g_y='AverageReturn',
                    g_z='Garage',
                    b_x='Iteration',
                    b_y='AverageReturn',
                    b_z='GarageWithModel',
                    trials=num_of_trials,
                    seeds=seeds,
                    plt_file=plt_file,
                    env_id=env_id,
                    x_label='Iteration',
                    y_label='AverageReturn')

            Rh.relplot(g_csvs=garage_csvs,
                       b_csvs=garage_with_model_csvs,
                       g_x='Iteration',
                       g_y='AverageReturn',
                       g_z='Garage',
                       b_x='Iteration',
                       b_y='AverageReturn',
                       b_z='GarageWithModel',
                       trials=num_of_trials,
                       seeds=seeds,
                       plt_file=relplt_file,
                       env_id=env_id,
                       x_label='Iteration',
                       y_label='AverageReturn')


def run_garage(env, seed, log_dir):
    '''
    Create garage model and training.

    Replace the ppo with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return:
    '''
    deterministic.set_seed(seed)

    with LocalTFRunner(snapshot_config) as runner:
        env = TfEnv(normalize(env))

        policy = CategoricalLSTMPolicy(env_spec=env.spec)

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=False,
                optimizer=FirstOrderOptimizer,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                    tf_optimizer_args=dict(learning_rate=1e-3),
                ),
            ),
        )

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=params['max_path_length'],
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
            ),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=0.02,
            center_adv=False,
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env)
        runner.train(n_epochs=params['n_epochs'], batch_size=2048)

        dowel_logger.remove_all()

        return tabular_log_file


def run_garage_with_model(env, seed, log_dir):
    '''
    Create garage model and training.

    Replace the ppo with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return:
    '''
    deterministic.set_seed(seed)

    with LocalTFRunner(snapshot_config) as runner:
        env = TfEnv(normalize(env))

        policy = CategoricalLSTMPolicyWithModel(env_spec=env.spec)

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=False,
                optimizer=FirstOrderOptimizer,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                    tf_optimizer_args=dict(learning_rate=1e-3),
                ),
            ),
        )

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=params['max_path_length'],
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
            ),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=0.02,
            center_adv=False,
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env)
        runner.train(n_epochs=params['n_epochs'], batch_size=2048)

        dowel_logger.remove_all()

        return tabular_log_file
