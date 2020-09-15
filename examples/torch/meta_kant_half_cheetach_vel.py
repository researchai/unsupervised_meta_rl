# diayn
# metakant

import altair as alt
import os
import click
import gym
import numpy as np
import pandas as pd
import torch
from altair_saver import save
from torch import nn
from torch.nn import functional as F

import garage.torch.utils as tu
from garage import wrap_experiment
from garage.envs import GarageEnv, DiaynEnvWrapper
from garage.envs import normalize
from garage.envs.mujoco import HalfCheetahVelEnv
from garage.experiment import deterministic, LocalRunner
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import EnvPoolSampler, SetTaskSampler
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.sampler import SkillWorker
from garage.sampler.local_skill_sampler import LocalSkillSampler
from garage.torch.algos import DIAYN
from garage.torch.algos import PEARL
from garage.torch.algos.discriminator import MLPDiscriminator
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import ContextConditionedPolicy
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.policies import TanhGaussianMLPSkillPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.q_functions import ContinuousMLPSkillQFunction

skills_num = 10

@wrap_experiment(snapshot_mode='none')
def diayn_half_cheetah_batch(ctxt=None, seed=1):
    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    env = GarageEnv(normalize(HalfCheetahVelEnv()))

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
                  max_path_length=300,
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
    runner.train(n_epochs=1000, batch_size=1000)  # 1000
    runner.save(999)  # saves the last episode

    return policy, diayn



env,
controller_policy,
skill_actor,
qf,
vf,
num_skills,
num_train_tasks,
num_test_tasks,
latent_dim,
encoder_hidden_sizes,\
tes                                                                                                                                                                                      t_env_sampler,
