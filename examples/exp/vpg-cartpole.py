"""Basic example of Experiment"""
import gym

from garage.baselines import LinearFeatureBaseline
from garage.contrib.exp import Experiment
from garage.contrib.torch.algos import VPG
from garage.contrib.exp.loggers import BasicLogger
from garage.contrib.exp.snapshotors import DiskSnapshotor
from garage.contrib.torch.policies import GaussianMLPPolicy

env = gym.make('CartPole-v1')

policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

agent = VPG(
    env_spec=env.spec,
    policy=policy,
    baseline=baseline,
    discount=0.99,
    optimizer_args = dict(tf_optimizer_args=dict(learning_rate=0.01, )))

# Alternatives: HDFS, S3, etc.
snapshotor = DiskSnapshotor(exp_dir='garage-vpg-cartpole')

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
    snapshotor=snapshotor,
    logger=logger,
    # exp variant
    n_itr=40,
    batch_size=4000,
    max_path_length=100,
)

exp.train()
