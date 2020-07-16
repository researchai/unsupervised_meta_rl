"""An example to test diayn written in PyTorch."""
import random

import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# import metaworld.benchmarks as mwb
import click

import garage.torch.utils as tu
from garage import wrap_experiment
from garage.envs import GarageEnv, DiaynEnvWrapper
from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import SkillWorker
from garage.sampler.local_skill_sampler import LocalSkillSampler
from garage.torch.algos import DIAYN
from garage.torch.algos.discriminator import MLPDiscriminator
from garage.torch.policies import TanhGaussianMLPSkillPolicy
from garage.torch.q_functions import ContinuousMLPSkillQFunction
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import EnvPoolSampler
from garage.sampler import LocalSampler
from garage.torch.algos import PEARL
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import ContextConditionedPolicy
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction

skills_num = 8
test_tasks_num = skills_num / 2

@wrap_experiment(snapshot_mode='none')
def diayn_half_cheetah_batch(ctxt=None, seed=1):

    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    env = GarageEnv(normalize(gym.make('HalfCheetah-v2')))

    policy = TanhGaussianMLPSkillPolicy(
        env_spec=env.spec,
        skills_num=skills_num,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPSkillQFunction(env_spec=env.spec,
                                      skills_num=skills_num,
                                      hidden_sizes=[256, 256],
                                      hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPSkillQFunction(env_spec=env.spec,
                                      skills_num=skills_num,
                                      hidden_sizes=[256, 256],
                                      hidden_nonlinearity=F.relu)

    discriminator = MLPDiscriminator(env_spec=env.spec,
                                     skills_num=skills_num,
                                     hidden_sizes=[64, 64],
                                     hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    diayn = DIAYN(env_spec=env.spec,
                  skills_num=skills_num,
                  discriminator=discriminator,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=1000,
                  max_path_length=500,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1e4,
                  recorded=True,  # enable the video recording func
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=256,
                  reward_scale=1.,
                  steps_per_epoch=1)

    if torch.cuda.is_available():
        tu.set_gpu_mode(True)
    else:
        tu.set_gpu_mode(False)
    diayn.to()
    worker_args = {"skills_num": skills_num}
    runner.setup(algo=diayn, env=env, sampler_cls=LocalSkillSampler,
                 worker_class=SkillWorker, worker_args=worker_args)
    runner.train(n_epochs=1000, batch_size=1000)

    return discriminator

@click.command()
@click.option('--num_epochs', default=1000)
@click.option('--num_train_tasks', default=skills_num)
@click.option('--num_test_tasks', default=test_tasks_num)
@click.option('--encoder_hidden_size', default=200)
@click.option('--net_size', default=300)
@click.option('--num_steps_per_epoch', default=4000)
@click.option('--num_initial_steps', default=4000)
@click.option('--num_steps_prior', default=750)
@click.option('--num_extra_rl_steps_posterior', default=750)
@click.option('--batch_size', default=256)
@click.option('--embedding_batch_size', default=64)
@click.option('--embedding_mini_batch_size', default=64)
@click.option('--max_path_length', default=150)
@wrap_experiment
def diayn_pearl_half_cheeth(
                            task_proposer,
                            ctxt=None,
                            seed=1,
                            num_epochs=1000,
                            num_train_tasks=skills_num,
                            num_test_tasks=skills_num,
                            latent_size=7,
                            encoder_hidden_size=200,
                            net_size=300,
                            meta_batch_size=16,
                            num_steps_per_epoch=4000,
                            num_initial_steps=4000,
                            num_tasks_sample=15,
                            num_steps_prior=750,
                            num_extra_rl_steps_posterior=750,
                            batch_size=256,
                            embedding_batch_size=64,
                            embedding_mini_batch_size=64,
                            max_path_length=150,
                            reward_scale=10.,
                            use_gpu=False):

    set_seed(seed)
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size)
    # create multi-task environment and sample tasks

    env = gym.make('HalfCheetah-v2')
    ML_train_envs = [
        GarageEnv(normalize(DiaynEnvWrapper(env, task_proposer, skills_num, task_name)))
        for task_name in range(skills_num)
    ]

    ML_test_envs = [
        GarageEnv(normalize(DiaynEnvWrapper(env, task_proposer, skills_num, task_name)))
        for task_name in random.sample(range(skills_num), test_tasks_num)
    ]

    env_sampler = EnvPoolSampler(ML_train_envs)
    env_sampler.grow_pool(num_train_tasks)
    env = env_sampler.sample(num_train_tasks)
    test_env_sampler = EnvPoolSampler(ML_test_envs)
    test_env_sampler.grow_pool(num_test_tasks)

    runner = LocalRunner(ctxt)

    # instantiate networks
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = PEARL.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    pearl = PEARL(
        env=env,
        policy_class=ContextConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_steps_per_epoch=num_steps_per_epoch,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_prior=num_steps_prior,
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        max_path_length=max_path_length,
        reward_scale=reward_scale,
    )

    tu.set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        pearl.to()

    runner.setup(algo=pearl,
                 env=env[0](),
                 sampler_cls=LocalSampler,
                 sampler_args=dict(max_path_length=max_path_length),
                 n_workers=1,
                 worker_class=PEARLWorker)

    runner.train(n_epochs=num_epochs, batch_size=batch_size)




s = np.random.randint(0, 1000)
task_proposer = diayn_half_cheetah_batch(seed=s)  # 521 in the sac_cheetah example
diayn_pearl_half_cheeth(task_proposer)

