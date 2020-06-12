"""An example to test diayn written in PyTorch."""
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler, SkillWorker
from garage.sampler.local_skill_sampler import LocalSkillSampler
from garage.torch.algos import SAC, DIAYN
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
import garage.torch.utils as tu


@wrap_experiment(snapshot_mode='none')
def sac_half_cheetah_batch(ctxt=None, seed=1):

    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    env = GarageEnv(normalize(gym.make('HalfCheetah-v2')))

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    discriminator = # TODO: implemented discriminator class

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    diayn = DIAYN(env_spec=env.spec,
                  skills_num=8,
                  discriminator=discriminator,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=1000,
                  max_path_length=500,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1e4,
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
    runner.setup(algo=diayn, env=env, sampler_cls=LocalSkillSampler, worker_class=SkillWorker)
    runner.train(n_epochs=1000, batch_size=1000)


s = np.random.randint(0, 1000)
sac_half_cheetah_batch(seed=521)
