"""PEARL benchmark script."""

import datetime
import os
import os.path as osp
import random

import akro
import dowel
from dowel import logger as dowel_logger
import numpy as np
import pytest
import torch
from torch.nn import functional as F  # NOQA

from garage.envs import normalize
from garage.envs.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.experiment import LocalRunner, run_experiment
from garage.sampler import InPlaceSampler
from garage.torch.algos import PEARLSAC
from garage.torch.embeddings import RecurrentEncoder
from garage.torch.modules import FlattenMLP, MLPEncoder
from garage.torch.policies import ContextConditionedPolicy, \
    TanhGaussianMLPPolicy
import garage.torch.utils as tu
from tests.fixtures import snapshot_config
import tests.helpers as Rh


# Hyperparams for baselines and garage
params = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=5, # dimension of the latent context vector
    net_size=300, # number of units per FC layer in each network
    path_to_weights=None, # path to pre-trained weights to load into networks
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
    ),
    algo_params=dict(
        meta_batch=4, # number of tasks to average the gradient across
        num_iterations=500, # number of data sampling / training iterates
        num_initial_steps=2000, # number of transitions collected per task before training
        num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=1000, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=1000, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=2000, # number of meta-gradient steps taken per iteration
        num_evals=2, # number of independent evals
        num_steps_per_eval=600,  # nuumber of transitions to eval on
        batch_size=256, # number of transitions in the RL batch
        embedding_batch_size=256, # number of transitions in the context batch
        embedding_mini_batch_size=256, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3e-4,
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        use_information_bottleneck=True, # False makes latent context deterministic
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=2, # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False, # recurrent or permutation-invariant encoder
        dump_eval_paths=False, # whether to save evaluation trajectories
    )
)


class TestBenchmarkPEARL:
    '''Compare benchmarks between garage and baselines.'''

    @pytest.mark.huge
    def test_benchmark_pearl(self):
        '''
        Compare benchmarks between garage and baselines.
        :return:
        '''
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = osp.join(os.getcwd(), 'data', 'local', 'benchmarks',
                                 'pearl', timestamp)
        result_json = {}
        
        env_id = 'half_cheetah_dir'
        task_dir = osp.join(benchmark_dir, env_id)
        plt_file = osp.join(benchmark_dir,
                            '{}_benchmark.png'.format(env_id))
        garage_csvs = []

        garage_dir = osp.join(task_dir, 'garage')

        garage_csv = run_garage(garage_dir)
        garage_csvs.append(garage_csv)


        Rh.plot(b_csvs=garage_csvs,
                g_csvs=garage_csvs,
                g_x='Epoch',
                g_y='AverageReturn',
                g_z='Garage',
                b_x='total/epochs',
                b_y='rollout/return',
                b_z='Baseline',
                trials=0,
                seeds=[1],
                plt_file=plt_file,
                env_id=env_id,
                x_label='Epoch',
                y_label='AverageReturn')

        result_json[env_id] = Rh.create_json(
            b_csvs=garage_csvs,
            g_csvs=garage_csvs,
            seeds=[1],
            trails=0,
            g_x='Epoch',
            g_y='AverageReturn',
            b_x='total/epochs',
            b_y='rollout/return',
            factor_g=params['steps_per_epoch'] * params['n_rollout_steps'],
            factor_b=1)

        Rh.write_file(result_json, 'PEARL')


def run_garage(log_dir):
    '''
    Create garage model and training.
    Replace the ddpg with the algorithm you want to run.
    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return:
    '''

    env = normalize(HalfCheetahDirEnv())
    runner = LocalRunner(snapshot_config)
    tasks = [0, 1]
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = params['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim \
        if params['algo_params']['use_next_obs_in_context'] \
            else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if params['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = params['net_size']
    recurrent = params['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MLPEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_dim=context_encoder_input_dim,
        output_dim=context_encoder_output_dim,
    )
    qf1 = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_dim=obs_dim + action_dim + latent_dim,
        output_dim=1,
    )
    qf2 = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_dim=obs_dim + action_dim + latent_dim,
        output_dim=1,
    )
    vf = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_dim=obs_dim + latent_dim,
        output_dim=1,
    )

    policy = TanhGaussianMLPPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    
    agent = ContextConditionedPolicy(
        latent_dim=latent_dim,
        context_encoder=context_encoder,
        policy=policy,
        use_ib=params['algo_params']['use_information_bottleneck'],
        use_next_obs=params['algo_params']['use_next_obs_in_context'],
    )

    pearlsac = PEARLSAC(
        env=env,
        train_tasks=list(tasks[:params['n_train_tasks']]),
        eval_tasks=list(tasks[-params['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **params['algo_params']
    )

    tu.set_gpu_mode(False)


    # Set up logger since we are not using run_experiment
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    tensorboard_log_dir = osp.join(log_dir)
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(tensorboard_log_dir))

    runner.setup(algo=pearlsac, env=env, sampler_cls=InPlaceSampler,
        sampler_args=dict(max_path_length=params['algo_params']['max_path_length']))
    runner.train(n_epochs=params['algo_params']['num_iterations'], batch_size=256)

    dowel_logger.remove_all()

    return tabular_log_file
