"""Basic example of Experiment"""
from garage.contrib.exp import Experiment
from garage.contrib.exp.agents.garage import VPG
from garage.contrib.exp.checkpointers import DiskCheckpointer
from garage.contrib.exp.envs.box2d import CartpoleEnv
from garage.contrib.exp.loggers import BasicLogger

# Immediate single step environment
# Alternatives: ReplayBuffer, ParallelSampler, etc.
env_variant = {}
observer = CartpoleEnv(**env_variant)

algo_variant = {}
agent = VPG(**algo_variant)

# Alternatives: HDFS, S3, etc.
checkpointer = DiskCheckpointer(exp_dir='garage-vpg-cartpole')

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
    checkpointer=checkpointer,
    logger=logger,
    # exp variant
    n_itr=40,
    batch_size=4000,
    max_path_length=100,
    discount=0.99,
)

exp.train()
