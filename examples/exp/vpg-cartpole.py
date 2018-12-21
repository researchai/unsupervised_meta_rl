"""Basic example of Experiment"""
from garage.contrib.exp import Experiment
from garage.contrib.exp.agents.garage.algos import VPG
from garage.contrib.exp.envs.box2d import CartpoleEnv
from garage.contrib.exp.loggers import BasicLogger
from garage.contrib.exp.snapshotors import DiskSnapshotor

# Immediate single step environment
# Alternatives: ReplayBuffer, ParallelSampler, etc.
env_variant = {}
observer = CartpoleEnv(**env_variant)

# Might also need to inject other components external to exp runner,
# e.g. policy & baseline. These components should also be wrapped up.
algo_variant = {}
agent = VPG(**algo_variant)

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
    observer=observer,
    agent=agent,
    snapshotor=snapshotor,
    logger=logger,
    # exp variant
    n_itr=40,
    batch_size=4000,
    max_path_length=100,
)

exp.train()
