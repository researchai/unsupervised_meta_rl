#!/usr/bin/env python3
"""
This is an example to train a task with PPO algorithm.

Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.

Results:
    AverageDiscountedReturn: 500
    RiseTime: itr 40

"""
import datetime
import multiprocessing
import sys
from datetime import datetime

import tensorflow as tf
from baselines import bench
from baselines import logger as baselines_logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.logger import configure
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
from garage import wrap_experiment
from garage.envs import GarageEnv, normalize_reward
from garage.envs.ml_wrapper import ML1WithPinnedGoal
from garage.envs.multi_task_metaworld_wrapper import MTMetaWorldWrapper
from garage.experiment.deterministic import set_seed

from tests.wrappers import AutoStopEnv

# env_id = 'push-v1'
# env_id = 'reach-v1'
# env_id = 'pick-place-v1'

@wrap_experiment
def trpo_baseline_ml1(ctxt=None, seed=1):
    """Run task."""
    set_seed(seed)
    max_path_length = 150

    Ml1_reach_envs = get_ML1_envs_test(env_id, max_path_length)
    env = MTMetaWorldWrapper(Ml1_reach_envs)
    env = AutoStopEnv(
        env=env,
        max_path_length=max_path_length)

    log_dir = 'data/local/experiment/trpo_baseline_ml1_'+datetime.now().strftime("%m_%d_%H_%M_%S")

    ncpu = max(multiprocessing.cpu_count() // 2, 1)
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.compat.v1.Session(config=config).__enter__()

    # Set up logger for baselines
    configure(dir=log_dir,
              format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    baselines_logger.info('rank {}: seed={}, logdir={}'.format(
        0, seed, baselines_logger.get_dir()))

    env = DummyVecEnv([
        lambda: bench.Monitor(
            env, baselines_logger.get_dir(), allow_early_resets=True)
    ])
    env = VecNormalize(env)

    set_global_seeds(seed)

    def policy_fn(name, ob_space, ac_space):
        """Create policy for baselines.

        Args:
            name (str): Policy name.
            ob_space (gym.spaces.Box) : Observation space.
            ac_space (gym.spaces.Box) : Action space.

        Returns:
            baselines.ppo1.mlp_policy: MLP policy for baselines.

        """
        return MlpPolicy(name=name,
                         ob_space=ob_space,
                         ac_space=ac_space,
                         hid_size=64,
                         num_hid_layers=2)

    nbatch = len(Ml1_reach_envs) * max_path_length
    timesteps = 6000000
    print(f'timesteps: {timesteps}, nbatch: {nbatch}')

    trpo_mpi.learn(env,
                   policy_fn,
                   timesteps_per_batch=nbatch,
                   max_kl=0.01,
                   cg_iters=10,
                   max_timesteps=timesteps,
                   gamma=0.99,
                   lam=0.97,
                   vf_iters=3,
                   vf_stepsize=0.01)


def get_ML1_envs_test(name, max_path_length):
    bench = ML1WithPinnedGoal.get_train_tasks(name)
    tasks = [{'task': 0, 'goal': i} for i in range(2)]
    ret = {}
    for task in tasks:
        new_bench = bench.clone(bench)
        new_bench.set_task(task)
        env = AutoStopEnv(
            env=GarageEnv(normalize_reward(new_bench.active_env)),
            max_path_length=max_path_length)
        env.spec.id = env_id
        ret[env_id+"_"+str(task['goal'])] = env
    return ret


# env_id = 'push-v1'
# env_id = 'reach-v1'
# env_id = 'pick-place-v1'

assert len(sys.argv) > 1

env_id = sys.argv[1]
s = int(sys.argv[2]) if len(sys.argv) > 2 else 0

trpo_baseline_ml1(seed=s)
