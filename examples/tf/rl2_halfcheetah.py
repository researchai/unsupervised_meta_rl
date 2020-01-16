import copy
from garage.envs import RL2Env
from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.envs.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO
from garage.tf.algos import RL2
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import GaussianGRUPolicy
from garage.sampler.rl2_sampler import RL2Sampler

# from metaworld.benchmarks import ML1
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def run_task(snapshot_config, *_):
    """Defines the main experiment routine.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): Configuration
            values for snapshotting.
        *_ (object): Hyperparameters (unused).

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        # env = RL2Env(env=HalfCheetahVelEnv())
        # env2 = RL2Env(env=HalfCheetahRandVelEnv())
        # env = RL2Env(env=HalfCheetahRandDirecEnv())
        # env = RL2Env(ML1.get_train_tasks('push-v1'))

        max_path_length = 100
        meta_batch_size = 200
        n_epochs = 500
        episode_per_task = 2
        env = HalfCheetahVelEnv()
        envs = [RL2Env(copy.deepcopy(env)) for _ in range(meta_batch_size)]
        tasks = envs[0].sample_tasks(meta_batch_size)
        for i, task in enumerate(tasks):
            envs[i].set_task(task)
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=64,
                                   env_spec=envs[0].spec,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=envs[0].spec)

        inner_algo = PPO(env_spec=envs[0].spec,
                         policy=policy,
                         baseline=baseline,
                         max_path_length=max_path_length * episode_per_task,
                         discount=0.99,
                         lr_clip_range=0.2,
                         meta_learn=True,
                         num_of_env=meta_batch_size,
                         episode_per_task=episode_per_task,
                         optimizer_args=dict(max_epochs=5))

        algo = RL2(policy=policy,
                   inner_algo=inner_algo,
                   max_path_length=max_path_length,
                   episode_per_task=episode_per_task,
                   normalize_adv=True,
                   positive_adv=False)

        runner.setup(algo,
                     envs,
                     sampler_cls=RL2Sampler,
                     sampler_args=dict(meta_batch_size=meta_batch_size,
                                       n_envs=meta_batch_size))
        runner.train(n_epochs=n_epochs,
                     batch_size=episode_per_task * max_path_length *
                     meta_batch_size)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
