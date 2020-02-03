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
import json
import copy

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
from garage.tf.algos import RL2
from garage.tf.algos import RL2PPO2
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianGRUPolicy
from garage.sampler import LocalSampler
from garage.sampler import RaySampler
from garage.sampler.rl2_worker import RL2Worker

from metaworld.benchmarks import ML1

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 0 : HalfCheetahVel
# 1 : HalfCheetahDir
# 2 : ML1-push
# 3 : ML1-reach
# 4 : ML1-pick-place
env_ind = 0
ML = env_ind in [2, 3, 4]

hyper_parameters = {
    'meta_batch_size': 50,
    'hidden_sizes': [64],
    'gae_lambda': 1,
    'discount': 0.99,
    'max_path_length': 150,
    'n_itr': 1000 if ML else 500, # total it will run [n_itr * steps_per_epoch] for garage
    'steps_per_epoch': 1,
    'rollout_per_task': 10,
    'positive_adv': False,
    'normalize_adv': True,
    'optimizer_lr': 1e-3,
    'lr_clip_range': 0.2,
    'optimizer_max_epochs': 5,
    'n_trials': 1,
    'n_test_tasks': 10,
    'cell_type': 'gru',
    'sampler_cls': RaySampler, 
    'use_all_workers': True
}

def _prepare_meta_env(env):
    if ML:
        task_samplers = task_sampler.EnvPoolSampler([RL2Env(env)])
        task_samplers.grow_pool(hyper_parameters['meta_batch_size'])
    else:
        task_samplers = task_sampler.SetTaskSampler(lambda: RL2Env(env()))
    return task_samplers.sample(1)[0](), task_samplers

class TestBenchmarkRL2:  # pylint: disable=too-few-public-methods
    """Compare benchmarks between garage and baselines."""

    @pytest.mark.huge
    def test_benchmark_rl2(self):  # pylint: disable=no-self-use
        """Compare benchmarks between garage and baselines."""
        if ML:
            if env_ind == 2:
                envs = [ML1.get_train_tasks('push-v1')]
                env_ids = ['ML1-push-v1']
            elif env_ind == 3:
                envs = [ML1.get_train_tasks('reach-v1')]
                env_ids = 'ML1-reach-v1'
            elif env_ind == 4:
                envs = [ML1.get_train_tasks('pick-place-v1')]
                env_ids = 'ML1-pick-place-v1'
            else:
                raise ValueError("Env index is wrong")
        else:
            if env_ind == 0:
                envs = [HalfCheetahVelEnv]
                env_ids = ['HalfCheetahVelEnv']
            elif env_ind == 1:
                envs = [HalfCheetahDirEnv]
                env_ids = ['HalfCheetahDirEnv']
            else:
                raise ValueError("Env index is wrong")

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/rl2/%s/' % timestamp
        result_json = {}
        for i, env in enumerate(envs):
            seeds = random.sample(range(100), hyper_parameters['n_trials'])
            task_dir = osp.join(benchmark_dir, env_ids[i])
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_ids[i]))
            garage_tf_csvs = []
            promp_csvs = []

            for trial in range(hyper_parameters['n_trials']):
                seed = seeds[trial]
                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                garage_tf_dir = trial_dir + '/garage'
                promp_dir = trial_dir + '/promp'

                with tf.Graph().as_default():
                    garage_tf_csv = run_garage(env, seed, garage_tf_dir)

                garage_tf_csvs.append(garage_tf_csv)

            with open(osp.join(garage_tf_dir, 'parameters.txt'), 'w') as outfile:
                hyper_parameters_copy = copy.deepcopy(hyper_parameters)
                hyper_parameters_copy['sampler_cls'] = str(hyper_parameters_copy['sampler_cls'])
                json.dump(hyper_parameters_copy, outfile)

            g_x = 'TotalEnvSteps'

            if ML:
                g_ys = [
                    'Evaluation/AverageReturn',
                    'Evaluation/SuccessRate',
                    'MetaTest/AverageReturn',
                    'MetaTest/SuccessRate'
                ]
            else:
                g_ys = [
                    'Evaluation/AverageReturn',
                    'MetaTest/AverageReturn'
                ]


            for g_y in g_ys:
                plt_file = osp.join(benchmark_dir,
                            '{}_benchmark_fit_after_linear_baseline_{}.png'.format(env_ids[i], g_y.replace('/', '-')))
                Rh.relplot(g_csvs=garage_tf_csvs,
                           b_csvs=None,
                           g_x=g_x,
                           g_y=g_y,
                           g_z='Garage',
                           b_x=None,
                           b_y=None,
                           b_z=None,
                           trials=hyper_parameters['n_trials'],
                           seeds=seeds,
                           plt_file=plt_file,
                           env_id=env_ids[i])


def run_garage(env, seed, log_dir):
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
                                     snapshot_mode='all',
                                     snapshot_gap=1)
    with LocalTFRunner(snapshot_config) as runner:
        env, task_samplers = _prepare_meta_env(env)

        policy = GaussianGRUPolicy(
            hidden_dims=hyper_parameters['hidden_sizes'],
            env_spec=env.spec,
            state_include_action=False)

        baseline = GarageLinearFeatureBaseline(env_spec=env.spec)

        inner_algo = RL2PPO2(
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
            ),
            meta_batch_size=hyper_parameters['meta_batch_size']
        )

        algo = RL2(
            policy=policy,
            inner_algo=inner_algo,
            max_path_length=hyper_parameters['max_path_length'],
            meta_batch_size=hyper_parameters['meta_batch_size'],
            task_sampler=task_samplers,
            steps_per_epoch=hyper_parameters['steps_per_epoch'])

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        text_log_file = osp.join(log_dir, 'debug.log')
        dowel_logger.add_output(dowel.TextOutput(text_log_file))
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo,
                     task_samplers.sample(hyper_parameters['meta_batch_size']),
                     sampler_cls=hyper_parameters['sampler_cls'],
                     n_workers=hyper_parameters['meta_batch_size'],
                     worker_class=RL2Worker,
                     sampler_args=dict(
                        use_all_workers=hyper_parameters['use_all_workers'],
                        n_paths_per_trial=hyper_parameters['rollout_per_task']))

        runner.setup_meta_evaluator(test_task_sampler=task_samplers,
                                    sampler_cls=hyper_parameters['sampler_cls'],
                                    n_test_tasks=hyper_parameters['n_test_tasks'])

        runner.train(n_epochs=hyper_parameters['n_itr'],
            batch_size=hyper_parameters['meta_batch_size'] * hyper_parameters['rollout_per_task'] * hyper_parameters['max_path_length'])

        dowel_logger.remove_all()

        return tabular_log_file
