import datetime
import os.path as osp
import random

import dowel
from dowel import logger as dowel_logger
import gym
import pytest
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import deterministic
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianConvBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalConvPolicy
from garage.tf.policies import CategoricalConvPolicyWithModel
from tests.fixtures import snapshot_config
import tests.helpers as Rh

num_of_trials = 3

params = {
    'conv_filters': (32, 64),
    'conv_filter_sizes': (8, 4),
    'conv_strides': (4, 2),
    'conv_pads': ('VALID', 'VALID'),
    'conv_pad': 'VALID',
    'hidden_sizes': (32, 32),
    'n_epochs': 1000,
    'batch_size': 2048,
    'use_trust_region': True
}


class TestBenchmarkPPO:
    '''Compare benchmarks between garage and baselines.'''

    @pytest.mark.huge
    def test_benchmark_ppo(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/ppo/%s/' % timestamp
        for env_id in ['CubeCrash-v0']:
            env = gym.make(env_id)

            seeds = random.sample(range(100), num_of_trials)

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark_ppo.png'.format(env_id))
            relplt_file = osp.join(benchmark_dir,
                                   '{}_benchmark_ppo_mean.png'.format(env_id))
            garage_csvs = []
            garage_model_csvs = []

            for trial in range(num_of_trials):
                seed = seeds[trial]

                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                garage_dir = trial_dir + '/garage'
                garage_model_dir = trial_dir + '/garage_with_models'

                with tf.Graph().as_default():
                    # Run garage algorithms
                    env.reset()
                    garage_csv = run_garage(env, seed, garage_dir)
                with tf.Graph().as_default():
                    # Run garage model algorithms
                    env.reset()
                    garage_model_csv = run_garage_model(
                        env, seed, garage_model_dir)

                garage_csvs.append(garage_csv)
                garage_model_csvs.append(garage_model_csv)

            env.close()

            Rh.relplot(g_csvs=garage_csvs,
                       b_csvs=garage_model_csvs,
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

            Rh.plot(g_csvs=garage_csvs,
                    b_csvs=garage_model_csvs,
                    g_x='Iteration',
                    g_y='AverageReturn',
                    b_x='Iteration',
                    b_y='AverageReturn',
                    trials=num_of_trials,
                    seeds=seeds,
                    plt_file=plt_file,
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
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=12,
                            inter_op_parallelism_threads=12)
    sess = tf.Session(config=config)

    with LocalTFRunner(snapshot_config, sess=sess, max_cpus=12) as runner:
        env = TfEnv(normalize(env))

        policy = CategoricalConvPolicy(
            env_spec=env.spec,
            conv_filters=params['conv_filters'],
            conv_filter_sizes=params['conv_filter_sizes'],
            conv_strides=params['conv_strides'],
            conv_pads=params['conv_pads'],
            hidden_sizes=params['hidden_sizes'])

        baseline = GaussianConvBaseline(
            env_spec=env.spec,
            regressor_args=dict(conv_filters=params['conv_filters'],
                                conv_filter_sizes=params['conv_filter_sizes'],
                                conv_strides=params['conv_strides'],
                                conv_pads=params['conv_pads'],
                                hidden_sizes=params['hidden_sizes'],
                                use_trust_region=params['use_trust_region']))

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            policy_ent_coeff=0.0,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                tf_optimizer_args=dict(learning_rate=1e-3),
            ),
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env)
        runner.train(n_epochs=params['n_epochs'],
                     batch_size=params['batch_size'])

        dowel_logger.remove_all()

        return tabular_log_file


def run_garage_model(env, seed, log_dir):
    '''
    Create garage model and training.

    Replace the ppo with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return:
    '''
    deterministic.set_seed(seed)
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=12,
                            inter_op_parallelism_threads=12)
    sess = tf.Session(config=config)

    with LocalTFRunner(snapshot_config, sess=sess, max_cpus=12) as runner:
        env = TfEnv(normalize(env))

        policy = CategoricalConvPolicyWithModel(
            env_spec=env.spec,
            conv_filters=params['conv_filters'],
            conv_filter_sizes=params['conv_filter_sizes'],
            conv_strides=params['conv_strides'],
            conv_pad=params['conv_pad'],
            hidden_sizes=params['hidden_sizes'])

        baseline = GaussianConvBaseline(
            env_spec=env.spec,
            regressor_args=dict(conv_filters=params['conv_filters'],
                                conv_filter_sizes=params['conv_filter_sizes'],
                                conv_strides=params['conv_strides'],
                                conv_pads=params['conv_pads'],
                                hidden_sizes=params['hidden_sizes'],
                                use_trust_region=params['use_trust_region']))

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            policy_ent_coeff=0.0,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                tf_optimizer_args=dict(learning_rate=1e-3),
            ),
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env)
        runner.train(n_epochs=params['n_epochs'],
                     batch_size=params['batch_size'])

        dowel_logger.remove_all()

        return tabular_log_file
