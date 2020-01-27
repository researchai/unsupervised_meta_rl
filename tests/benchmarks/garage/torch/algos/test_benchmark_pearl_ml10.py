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
#pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
from metaworld.benchmarks import ML10

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.envs.env_spec import EnvSpec
from garage.experiment import deterministic, LocalRunner, run_experiment
from garage.experiment.snapshotter import SnapshotConfig
from garage.sampler import PEARLSampler
from garage.torch.algos import PEARLSAC
from garage.torch.embeddings import MLPEncoder
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.policies import ContextConditionedPolicy, \
    TanhGaussianMLPPolicy
import garage.torch.utils as tu
from tests import benchmark_helper
import tests.helpers as Rh

# hyperparams for baselines and garage
params = dict(
    num_epochs=250,
    num_train_tasks=10,
    num_test_tasks=5,
    latent_size=7, # dimension of the latent context vector
    net_size=300, # number of units per FC layer in each network
    env_params=dict(
        n_tasks=15, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
    ),
    algo_params=dict(
        meta_batch_size=16, # number of tasks to average the gradient across
        num_steps_per_epoch=4000, # number of data sampling / training iterates
        num_initial_steps=4000, # number of transitions collected per task before training
        num_tasks_sample=15, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=750, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=750, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_evals=5, # number of independent evals
        num_steps_per_eval=450,  # nuumber of transitions to eval on
        batch_size=256, # number of transitions in the RL batch
        embedding_batch_size=100, # number of transitions in the context batch
        embedding_mini_batch_size=100, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=150, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3E-4,
        reward_scale=10., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=2, # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False, # recurrent or permutation-invariant encoder
        use_information_bottleneck=True, # False makes latent context deterministic
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
    ),
    n_trials=3,
    use_gpu=True,
)


class TestBenchmarkPEARL:
    '''Compare benchmarks between garage and baselines.'''

    @pytest.mark.huge
    def test_benchmark_pearl(self):
        '''
        Compare benchmarks between garage and baselines.
        :return:
        '''
        envs = [ML10.get_train_tasks()]
        test_env = ML10.get_test_tasks()
        env_ids = ['ML10']
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = osp.join(os.getcwd(), 'data', 'local', 'benchmarks',
                                 'pearl', timestamp)
        result_json = {}

        for idx in range(1):
            env = envs[idx]
            env_id = env_ids[idx]
            seeds = random.sample(range(100), params['n_trials'])
            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))
            garage_csvs = []
            pearl_csvs = []

            for trial in range(params['n_trials']):
                seed = seeds[trial]
                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                garage_dir = trial_dir + '/garage'
                #pearl_dir = trial_dir + '/pearl'

                print(test_env)
                garage_csv = run_garage(env, seed, garage_dir, test_env=test_env)
                #pearl_csv = run_pearl(env, seed, garage_dir)

                garage_csvs.append(garage_csv)
                #pearl_csvs.append(pearl_csv)
            
            env.close()

            benchmark_helper.plot_average_over_trials(
                [garage_csvs],
                ys=['TestAverageReturn'],
                plt_file=plt_file,
                env_id=env_id,
                x_label='TotalEnvSteps',
                y_label='TestTaskAverageReturn',
                names=['garage_pearl'],
            )

            factor_val = params['algo_params']['meta_batch_size'] * params['algo_params']['max_path_length']
            result_json[env_id] = benchmark_helper.create_json(
                [garage_csvs],
                seeds=seeds,
                trials=params['n_trials'],
                xs=['TotalEnvSteps'],
                ys=['TestAverageReturn'],
                factors=[factor_val],
                names=['garage_pearl'])

            Rh.write_file(result_json, 'PEARL')


def run_garage(env, seed, log_dir, test_env=None):
    '''
    Create garage model and training.
    Replace the ddpg with the algorithm you want to run.
    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return:
    '''
    deterministic.set_seed(seed)
    env = normalize(env)
    test_env = normalize(test_env)
    path = os.path.join(os.getcwd(), 'data/local/experiment')
    snapshot_config = SnapshotConfig(snapshot_dir=path,
                                     snapshot_mode='gap',
                                     snapshot_gap=10)
    runner = LocalRunner(snapshot_config)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = params['latent_size']
    encoder_in_dim = 2 * obs_dim + action_dim + reward_dim \
        if params['algo_params']['use_next_obs_in_context'] \
            else obs_dim + action_dim + reward_dim
    encoder_out_dim = latent_dim * 2 if params['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = params['net_size']

    context_encoder = MLPEncoder(input_dim=encoder_in_dim,
                                 output_dim=encoder_out_dim,
                                 hidden_sizes=[200, 200, 200])

    space_a = akro.Box(low=-1, high=1, shape=(obs_dim+latent_dim, ), dtype=np.float32)
    space_b = akro.Box(low=-1, high=1, shape=(action_dim, ), dtype=np.float32)
    qf_env = EnvSpec(space_a, space_b)

    qf1 = ContinuousMLPQFunction(env_spec=qf_env,
                                 hidden_sizes=[net_size, net_size, net_size])

    qf2 = ContinuousMLPQFunction(env_spec=qf_env,
                                 hidden_sizes=[net_size, net_size, net_size])

    obs_space = akro.Box(low=-1, high=1, shape=(obs_dim, ), dtype=np.float32)
    action_space = akro.Box(low=-1, high=1, shape=(latent_dim, ), dtype=np.float32)
    vf_env = EnvSpec(obs_space, action_space)

    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    policy = TanhGaussianMLPPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )

    context_conditioned_policy = ContextConditionedPolicy(
        latent_dim=latent_dim,
        context_encoder=context_encoder,
        policy=policy,
        use_ib=params['algo_params']['use_information_bottleneck'],
        use_next_obs=params['algo_params']['use_next_obs_in_context'],
    )

    pearlsac = PEARLSAC(
        env=env,
        test_env=test_env,
        policy=context_conditioned_policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        num_train_tasks=params['num_train_tasks'],
        num_test_tasks=params['num_test_tasks'],
        latent_dim=latent_dim,
        **params['algo_params']
    )

    tu.set_gpu_mode(params['use_gpu'])
    if params['use_gpu'] == True: 
        pearlsac.to()

    # Set up logger since we are not using run_experiment
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    tensorboard_log_dir = osp.join(log_dir)
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(tensorboard_log_dir))

    runner.setup(algo=pearlsac, env=env, sampler_cls=PEARLSampler,
        sampler_args=dict(max_path_length=params['algo_params']['max_path_length']))
    runner.train(n_epochs=params['num_epochs'], batch_size=256)

    dowel_logger.remove_all()

    return tabular_log_file