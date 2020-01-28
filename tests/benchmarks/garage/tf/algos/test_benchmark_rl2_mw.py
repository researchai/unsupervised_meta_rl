"""This script creates a regression test over garage-TRPO and baselines-TRPO.

Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length. We also need to change the
garage.tf.samplers.BatchSampler to smooth the reward curve.
"""
import datetime
import os.path as osp
import random
import numpy as np

import dowel
from dowel import logger as dowel_logger
import pytest
import tensorflow as tf

from garage.experiment import deterministic
from tests.fixtures import snapshot_config
import tests.helpers as Rh

from garage.envs import RL2Env
from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.envs.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.experiment import task_sampler
from garage.experiment.snapshotter import SnapshotConfig
from garage.np.baselines import LinearFeatureBaseline as GarageLinearFeatureBaseline
from garage.tf.algos import PPO as GaragePPO
from garage.tf.algos import RL2
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianLSTMPolicy
from garage.sampler import LocalSampler
from garage.sampler.rl2_sampler import RL2Sampler
from garage.sampler.rl2_worker import RL2Worker

from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from maml_zoo.envs.rl2_env import rl2env
from maml_zoo.algos.ppo import PPO
from maml_zoo.trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.rl2_sample_processor import RL2SampleProcessor
from maml_zoo.policies.gaussian_rnn_policy import GaussianRNNPolicy
from maml_zoo.logger import logger

from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_CLS_DICT
ML10_ARGS = MEDIUM_MODE_ARGS_KWARGS
ML10_ENVS = MEDIUM_MODE_CLS_DICT
ML45_ARGS = HARD_MODE_ARGS_KWARGS
ML45_ENVS = HARD_MODE_CLS_DICT

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

hyper_parameters = {
    'meta_batch_size': 50,
    'hidden_sizes': [64],
    'gae_lambda': 1,
    'discount': 0.99,
    'max_path_length': 150,
    'n_itr': 2,
    'rollout_per_task': 10,
    'positive_adv': False,
    'normalize_adv': True,
    'optimizer_lr': 1e-3,
    'lr_clip_range': 0.2,
    'optimizer_max_epochs': 5,
    'n_trials': 1,
    'cell_type': 'lstm'
}

# True if ML10, false if ML45
ML10 = True

class TestBenchmarkRL2:  # pylint: disable=too-few-public-methods
    """Compare benchmarks between garage and baselines."""

    @pytest.mark.huge
    def test_benchmark_rl2(self):  # pylint: disable=no-self-use
        """Compare benchmarks between garage and baselines."""
        if ML10:
            env_id = 'ML10'
            ML_train_envs = [
                RL2Env(env(*ML10_ARGS['train'][task]['args'],
                    **ML10_ARGS['train'][task]['kwargs']))
                for (task, env) in ML10_ENVS['train'].items()
            ]
        else:
            env_obs_dim = [env().observation_space.shape[0] for (_, env) in ML45_ENVS['train'].items()]
            max_obs_dim = max(env_obs_dim)
            env_id = 'ML45'
            ML_train_envs = [
                RL2Env(env(*ML45_ARGS['train'][task]['args'],
                    **ML45_ARGS['train'][task]['kwargs']), max_obs_dim)
                for (task, env) in ML45_ENVS['train'].items()
            ]
        tasks = task_sampler.EnvPoolSampler(ML_train_envs)
        tasks.grow_pool(hyper_parameters['meta_batch_size'])
        envs = tasks.sample(hyper_parameters['meta_batch_size'])
        env = envs[0]()

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/rl2/%s/' % timestamp
        result_json = {}

        # Start main loop
        seeds = random.sample(range(100), hyper_parameters['n_trials'])
        task_dir = osp.join(benchmark_dir, env_id)
        garage_tf_csvs = []

        for trial in range(hyper_parameters['n_trials']):
            seed = seeds[trial]
            trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
            garage_tf_dir = trial_dir + '/garage'

            with tf.Graph().as_default():
                env.reset()
                garage_tf_csv = run_garage(env, envs, seed, garage_tf_dir)

            garage_tf_csvs.append(garage_tf_csv)


        g_x = 'TotalEnvSteps'
        g_y1 = 'Evaluation/AverageReturn'
        g_y2 = 'SuccessRate'

        plt_file1 = osp.join(benchmark_dir,
                            '{}_benchmark_average_return.png'.format(env_id))
        plt_file2 = osp.join(benchmark_dir,
                            '{}_benchmark_success_rate.png'.format(env_id))

        Rh.relplot(g_csvs=garage_tf_csvs,
                   b_csvs=None,
                   g_x=g_x,
                   g_y=g_y1,
                   g_z='Garage',
                   b_x=None,
                   b_y=None,
                   b_z='ProMP',
                   trials=hyper_parameters['n_trials'],
                   seeds=seeds,
                   plt_file=plt_file1,
                   env_id=env_id,
                   x_label=g_x,
                   y_label=g_y1)

        Rh.relplot(g_csvs=garage_tf_csvs,
                   b_csvs=None,
                   g_x=g_x,
                   g_y=g_y2,
                   g_z='Garage',
                   b_x=None,
                   b_y=None,
                   b_z='ProMP',
                   trials=hyper_parameters['n_trials'],
                   seeds=seeds,
                   plt_file=plt_file2,
                   env_id=env_id,
                   x_label=g_x,
                   y_label=g_y2)


def run_garage(env, envs, seed, log_dir):
    """Create garage Tensorflow PPO model and training.

    Args:
        env (dict): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    deterministic.set_seed(seed)
    snapshot_config = SnapshotConfig(snapshot_dir=log_dir,
                                     snapshot_mode='gap',
                                     snapshot_gap=10)
    with LocalTFRunner(snapshot_config) as runner:
        policy = GaussianLSTMPolicy(
            hidden_dim=hyper_parameters['hidden_sizes'][0],
            env_spec=env.spec,
            state_include_action=False)

        baseline = GarageLinearFeatureBaseline(env_spec=env.spec)

        inner_algo = GaragePPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=hyper_parameters['max_path_length'] * hyper_parameters['rollout_per_task'],
            discount=hyper_parameters['discount'],
            gae_lambda=hyper_parameters['gae_lambda'],
            lr_clip_range=hyper_parameters['lr_clip_range'],
            optimizer_args=dict(
                max_epochs=hyper_parameters['optimizer_max_epochs'],
                tf_optimizer_args=dict(
                    learning_rate=hyper_parameters['optimizer_lr'],
                ),
            )
        )

        algo = RL2(
            policy=policy,
            inner_algo=inner_algo,
            max_path_length=hyper_parameters['max_path_length'])

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(
            algo,
            envs,
            sampler_cls=LocalSampler,
            n_workers=hyper_parameters['meta_batch_size'],
            worker_class=RL2Worker)

        runner.train(n_epochs=hyper_parameters['n_itr'],
            batch_size=hyper_parameters['meta_batch_size'] * hyper_parameters['rollout_per_task'] * hyper_parameters['max_path_length'])

        dowel_logger.remove_all()

        return tabular_log_file