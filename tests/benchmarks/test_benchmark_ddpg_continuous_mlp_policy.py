'''
This script creates a regression test over garage-DDPG and baselines-DDPG.
It get Mujoco1M benchmarks from baselines benchmark, and test each task in
its trial times on garage model and baselines model. For each task, there will
be `trial` times with different random seeds. For each trial, there will be two
log directories corresponding to baselines and garage. And there will be a plot
plotting the average return curve from baselines and garage.
'''
import datetime
import os
import os.path as osp
import random

from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common.misc_util import set_global_seeds
from baselines.ddpg.memory import Memory
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
import baselines.ddpg.training as training
from baselines.logger import configure
import dowel
from dowel import logger as dowel_logger
import gym
from mpi4py import MPI
import numpy as np
import pytest
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import deterministic
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.algos import DDPGold
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.policies import ContinuousMLPPolicyWithModel
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import snapshot_config
import tests.helpers as Rh
from tests.wrappers import AutoStopEnv

# Hyperparams for baselines and garage
params = {
    'policy_lr': 1e-4,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [64, 64],
    'qf_hidden_sizes': [64, 64],
    'n_epochs': 300,
    'n_epoch_cycles': 20,
    'n_rollout_steps': 100,
    'n_train_steps': 50,
    'discount': 0.9,
    'tau': 1e-2,
    'replay_buffer_size': int(1e6),
    'sigma': 0.2,
}


class TestBenchmarkDDPG:
    '''Compare benchmarks between garage and baselines.'''

    @pytest.mark.huge
    def test_benchmark_ddpg(self):
        '''
        Compare benchmarks between garage and baselines.
        :return:
        '''
        # Load Mujoco1M tasks, you can check other benchmarks here
        # https://github.com/openai/baselines/blob/master/baselines/bench/benchmarks.py
        mujoco1m = benchmarks.get_benchmark('Mujoco1M')

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = osp.join(os.getcwd(), 'data', 'local', 'benchmarks',
                                 'ddpg', timestamp)
        result_json = {}
        for task in mujoco1m['tasks']:
            env_id = task['env_id']
            env = gym.make(env_id)

            seeds = random.sample(range(100), 3)

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(
                benchmark_dir,
                '{}_benchmark_ddpg_continuous_mlp_policy.png'.format(env_id))
            garage_models_csvs = []
            garage_csvs = []

            for trial in range(3):
                seed = seeds[trial]

                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                garage_dir = trial_dir + '/garage'
                garage_models_dir = trial_dir + '/garage_models'

                with tf.Graph().as_default():
                    env.reset()
                    garage_csv = run_garage(env, seed, garage_dir)
                with tf.Graph().as_default():
                    env.reset()
                    garage_models_csv = run_garage_with_models(
                        env, seed, garage_models_dir)
                garage_csvs.append(garage_csv)
                garage_models_csvs.append(garage_models_csv)

            env.close()

            Rh.plot(b_csvs=garage_models_csvs,
                    g_csvs=garage_csvs,
                    g_x='Epoch',
                    g_y='AverageReturn',
                    b_x='Epoch',
                    b_y='AverageReturn',
                    trials=3,
                    seeds=seeds,
                    plt_file=plt_file,
                    env_id=env_id,
                    x_label='Iteration',
                    y_label='AverageReturn')

            result_json[env_id] = Rh.create_json(b_csvs=garage_models_csvs,
                                                 g_csvs=garage_csvs,
                                                 seeds=seeds,
                                                 trails=3,
                                                 g_x='Epoch',
                                                 g_y='AverageReturn',
                                                 b_x='Epoch',
                                                 b_y='AverageReturn',
                                                 factor_g=2048,
                                                 factor_b=2048)

        Rh.write_file(result_json, 'DDPG')


def run_garage(env, seed, log_dir):
    '''
    Create garage model and training.
    Replace the ddpg with the algorithm you want to run.
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
        # Set up params for ddpg
        action_noise = OUStrategy(env.spec, sigma=params['sigma'])

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=params['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=params['qf_hidden_sizes'],
                                    hidden_nonlinearity=tf.nn.relu)

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=params['replay_buffer_size'],
            time_horizon=params['n_rollout_steps'])

        ddpg = DDPGold(env_spec=env.spec,
                       policy=policy,
                       qf=qf,
                       replay_buffer=replay_buffer,
                       policy_lr=params['policy_lr'],
                       qf_lr=params['qf_lr'],
                       target_update_tau=params['tau'],
                       n_train_steps=params['n_train_steps'],
                       discount=params['discount'],
                       min_buffer_size=int(1e4),
                       exploration_strategy=action_noise,
                       policy_optimizer=tf.train.AdamOptimizer,
                       qf_optimizer=tf.train.AdamOptimizer)

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(ddpg, env, sampler_args=dict(n_envs=12))
        runner.train(n_epochs=params['n_epochs'],
                     n_epoch_cycles=params['n_epoch_cycles'],
                     batch_size=params['n_rollout_steps'])

        dowel_logger.remove_all()

        return tabular_log_file


def run_garage_with_models(env, seed, log_dir):
    '''
    Create garage model and training.
    Replace the ddpg with the algorithm you want to run.
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
        # Set up params for ddpg
        action_noise = OUStrategy(env.spec, sigma=params['sigma'])

        policy = ContinuousMLPPolicyWithModel(
            env_spec=env.spec,
            name='ContinuousMLPPolicyWithModel',
            hidden_sizes=params['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=params['qf_hidden_sizes'],
                                    hidden_nonlinearity=tf.nn.relu,
                                    name='ContinuousMLPQFunction2')

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=params['replay_buffer_size'],
            time_horizon=params['n_rollout_steps'])

        ddpg = DDPG(env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    policy_lr=params['policy_lr'],
                    qf_lr=params['qf_lr'],
                    target_update_tau=params['tau'],
                    n_train_steps=params['n_train_steps'],
                    discount=params['discount'],
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    policy_optimizer=tf.train.AdamOptimizer,
                    qf_optimizer=tf.train.AdamOptimizer)

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(ddpg, env, sampler_args=dict(n_envs=12))
        runner.train(n_epochs=params['n_epochs'],
                     n_epoch_cycles=params['n_epoch_cycles'],
                     batch_size=params['n_rollout_steps'])

        dowel_logger.remove_all()

        return tabular_log_file
