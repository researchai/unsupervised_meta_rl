"""Basic example of Experiment"""
import gym

from garage.contrib.exp import Experiment
from garage.contrib.torch.algos import VPG
from garage.contrib.exp.loggers import BasicLogger
from garage.contrib.exp.checkpointer import DiskCheckpointer
from garage.contrib.torch.policies import GaussianMLPPolicy
from garage.contrib.exp.core.misc import get_env_spec


env = gym.make('Pendulum-v0')
env_spec = get_env_spec(env)

policy = GaussianMLPPolicy(env_spec=env_spec, hidden_sizes=(32, 32))

# baseline = LinearFeatureBaseline(env_spec=env_spec)

agent = VPG(
    env_spec=env_spec,
    policy=policy,
    # baseline=baseline,
    discount=0.99)

# Alternatives: HDFS, S3, etc.
snapshotor = DiskCheckpointer(exp_dir='garage-vpg-cartpole')

# Alternativs: Tensorboard, Plotter
logger = BasicLogger()
"""
Initialize or load checkpoint from exp_dir.

/exp_dir
    /checkpoint
        prefix-policy.pkl
        prefix-algo.pkl
        prefix-env.pkl
        prefix-replaybuffer.pkl
    /logs
        prefix-summary.log
        prefix-info.log
"""
exp = Experiment(
    env=env,
    agent=agent,
    checkpointer=snapshotor,
    logger=logger,
    # exp variant
    n_itr=40,
    batch_size=4000,
    max_path_length=100,
)

exp.train()
