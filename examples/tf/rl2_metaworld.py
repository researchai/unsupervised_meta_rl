import copy
from garage.envs import RL2Env
from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.experiment import task_sampler
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
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_CLS_DICT
ML10_ARGS = MEDIUM_MODE_ARGS_KWARGS
ML10_ENVS = MEDIUM_MODE_CLS_DICT

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

        ML10_train_envs = [
            env(*ML10_ARGS['train'][task]['args'],
                **ML10_ARGS['train'][task]['kwargs'])
            for (task, env) in ML10_ENVS['train'].items()
        ]
        tasks = task_sampler.EnvPoolSampler(ML10_train_envs)
        assert tasks.n_tasks == 10
        updates = tasks.sample(10)
        for env in ML10_train_envs:
            assert any(env is update() for update in updates)

        import pdb
        pdb.set_trace()
        max_path_length = 100
        meta_batch_size = 200
        n_epochs = 500
        episode_per_task = 10
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=64,
                                   env_spec=env.spec,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        inner_algo = PPO(env_spec=env.spec,
                         policy=policy,
                         baseline=baseline,
                         max_path_length=max_path_length * episode_per_task,
                         discount=0.99,
                         lr_clip_range=0.2,
                         optimizer_args=dict(max_epochs=5))

        algo = RL2(policy=policy,
                   inner_algo=inner_algo,
                   max_path_length=max_path_length)

        runner.setup(algo,
                     env,
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
