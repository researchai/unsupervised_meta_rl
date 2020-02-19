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
from garage.envs import normalize_reward
from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.envs.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.experiment import task_sampler
from garage.experiment.snapshotter import SnapshotConfig
from garage.np.baselines import LinearFeatureBaseline as GarageLinearFeatureBaseline
from garage.tf.algos import RL2
from garage.tf.algos import RL2PPO
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianGRUPolicy
from garage.sampler import LocalSampler
from garage.sampler import RaySampler
from garage.sampler.rl2_worker import RL2Worker
from garage.envs import TaskIdWrapper

from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_CLS_DICT

ML10_ARGS = MEDIUM_MODE_ARGS_KWARGS
ML10_ENVS = MEDIUM_MODE_CLS_DICT


hyper_parameters = {
    'meta_batch_size': 40,
    'hidden_sizes': [200, 200],
    'gae_lambda': 1,
    'discount': 0.99,
    'max_path_length': 150,
    'n_itr': 1500,
    'rollout_per_task': 10,
    'test_rollout_per_task': 2,
    'positive_adv': False,
    'normalize_adv': True,
    'optimizer_lr': 1e-3,
    'lr_clip_range': 0.2,
    'optimizer_max_epochs': 5,
    'n_trials': 1,
    'n_test_tasks': 5,
    'cell_type': 'gru',
    'sampler_cls': RaySampler,
    'use_all_workers': True
}

class TestBenchmarkRL2:  # pylint: disable=too-few-public-methods
    """Compare benchmarks between garage and baselines."""

    @pytest.mark.huge
    def test_benchmark_rl2(self):  # pylint: disable=no-self-use
        """Compare benchmarks between garage and baselines."""
        env_id = 'ML10'
        ML_train_envs = [
            TaskIdWrapper(RL2Env(env(*ML10_ARGS['train'][task]['args'],
                **ML10_ARGS['train'][task]['kwargs'])), task_id=task_id, task_name=task)
            for (task_id, (task, env)) in enumerate(ML10_ENVS['train'].items())
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
                garage_tf_csv = run_garage(env, envs, tasks, seed, garage_tf_dir)

            garage_tf_csvs.append(garage_tf_csv)

        with open(osp.join(garage_tf_dir, 'parameters.txt'), 'w') as outfile:
            hyper_parameters_copy = copy.deepcopy(hyper_parameters)
            hyper_parameters_copy['sampler_cls'] = str(hyper_parameters_copy['sampler_cls'])
            json.dump(hyper_parameters_copy, outfile)

        g_x = 'TotalEnvSteps'
        g_ys = [
            'Evaluation/AverageReturn',
            'Evaluation/SuccessRate',
        ]

        for g_y in g_ys:
            plt_file = osp.join(benchmark_dir,
                            '{}_benchmark_max_entropy_{}.png'.format(env_id, g_y.replace('/', '-')))
            Rh.relplot(g_csvs=garage_tf_csvs,
                       b_csvs=None,
                       g_x=g_x,
                       g_y=g_y,
                       g_z='Garage',
                       b_x=None,
                       b_y=None,
                       b_z='ProMP',
                       trials=hyper_parameters['n_trials'],
                       seeds=seeds,
                       plt_file=plt_file,
                       env_id=env_id,
                       x_label=g_x,
                       y_label=g_y)


def run_garage(env, envs, tasks, seed, log_dir):
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
        policy = GaussianGRUPolicy(
            hidden_dims=hyper_parameters['hidden_sizes'],
            env_spec=env.spec,
            state_include_action=False)

        baseline = GarageLinearFeatureBaseline(env_spec=env.spec)

        inner_algo = RL2PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=hyper_parameters['max_path_length'] * hyper_parameters['rollout_per_task'],
            discount=hyper_parameters['discount'],
            gae_lambda=hyper_parameters['gae_lambda'],
            lr_clip_range=hyper_parameters['lr_clip_range'],
            entropy_method='max',
            center_adv=False,
            stop_entropy_gradient=True,
            policy_ent_coeff=0.1,
            optimizer_args=dict(
                max_epochs=hyper_parameters['optimizer_max_epochs'],
                tf_optimizer_args=dict(
                    learning_rate=hyper_parameters['optimizer_lr'],
                ),
            )
        )

        # Need to pass this if meta_batch_size < num_of_tasks
        task_names = list(ML10_ENVS['train'].keys())
        algo = RL2(
            policy=policy,
            inner_algo=inner_algo,
            max_path_length=hyper_parameters['max_path_length'],
            meta_batch_size=hyper_parameters['meta_batch_size'],
            task_sampler=tasks,
            task_names=None if hyper_parameters['meta_batch_size'] >= len(task_names) else task_names)

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        text_log_file = osp.join(log_dir, 'debug.log')
        dowel_logger.add_output(dowel.TextOutput(text_log_file))
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(
            algo,
            envs,
            sampler_cls=hyper_parameters['sampler_cls'],
            n_workers=hyper_parameters['meta_batch_size'],
            worker_class=RL2Worker,
            sampler_args=dict(
                use_all_workers=hyper_parameters['use_all_workers']),
            worker_args=dict(
                n_paths_per_trial=hyper_parameters['rollout_per_task']))

        # meta evaluator
        ML_test_envs = [
            TaskIdWrapper(RL2Env(env(*ML10_ARGS['test'][task]['args'],
                **ML10_ARGS['test'][task]['kwargs'])), task_id=task_id, task_name=task)
            for (task_id, (task, env)) in enumerate(ML10_ENVS['test'].items())
        ]
        test_tasks = task_sampler.EnvPoolSampler(ML_test_envs)
        test_tasks.grow_pool(hyper_parameters['n_test_tasks'])

        test_task_names = list(ML10_ENVS['test'].keys())

        runner.setup_meta_evaluator(test_task_sampler=test_tasks,
                                    n_exploration_traj=hyper_parameters['rollout_per_task'],
                                    n_test_rollouts=hyper_parameters['test_rollout_per_task'],
                                    n_test_tasks=hyper_parameters['n_test_tasks'],
                                    n_workers=hyper_parameters['n_test_tasks'],
                                    test_task_names=None if hyper_parameters['n_test_tasks'] >= len(test_task_names) else test_task_names)

        runner.train(n_epochs=hyper_parameters['n_itr'],
            batch_size=hyper_parameters['meta_batch_size'] * hyper_parameters['rollout_per_task'] * hyper_parameters['max_path_length'])

        dowel_logger.remove_all()

        return tabular_log_file
