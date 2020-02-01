import copy
from garage.envs import RL2Env
from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.experiment import task_sampler
from garage.envs.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import RL2PPO
from garage.tf.algos import RL2
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import GaussianGRUPolicy
from garage.sampler import LocalSampler
from garage.sampler.rl2_worker import RL2Worker
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_CLS_DICT
ML10_ARGS = MEDIUM_MODE_ARGS_KWARGS
ML10_ENVS = MEDIUM_MODE_CLS_DICT

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
        max_path_length = 150
        meta_batch_size = 50
        n_epochs = 500
        episode_per_task = 10
        steps_per_epoch = 1
        n_test_tasks = 1

        ML10_train_envs = [
            RL2Env(env(*ML10_ARGS['train'][task]['args'],
                **ML10_ARGS['train'][task]['kwargs']))
            for (task, env) in ML10_ENVS['train'].items()
        ]
        tasks = task_sampler.EnvPoolSampler(ML10_train_envs)
        tasks.grow_pool(meta_batch_size)
        envs = tasks.sample(meta_batch_size)
        env = envs[0]()
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dims=[64],
                                   env_spec=env.spec,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        inner_algo = RL2PPO(env_spec=env.spec,
                         policy=policy,
                         baseline=baseline,
                         max_path_length=max_path_length * episode_per_task,
                         discount=0.99,
                         gae_lambda=0.95,
                         lr_clip_range=0.2,
                         optimizer_args=dict(
                            batch_size=32,
                            max_epochs=10,
                         ),
                         stop_entropy_gradient=True,
                         entropy_method='max',
                         policy_ent_coeff=0.02,
                         center_adv=False)

        algo = RL2(policy=policy,
                   inner_algo=inner_algo,
                   max_path_length=max_path_length,
                   meta_batch_size=meta_batch_size,
                   task_sampler=tasks,
                   steps_per_epoch=steps_per_epoch)

        runner.setup(algo,
                     envs,
                     sampler_cls=LocalSampler,
                     n_workers=meta_batch_size,
                     worker_class=RL2Worker)

        ML10_test_envs = [
            RL2Env(env(*ML10_ARGS['test'][task]['args'],
                **ML10_ARGS['test'][task]['kwargs']))
            for (task, env) in ML10_ENVS['test'].items()
        ]
        test_tasks = task_sampler.EnvPoolSampler(ML10_test_envs)
        runner.setup_meta_evaluator(test_task_sampler=test_tasks,
                                    sampler_cls=LocalSampler,
                                    n_test_tasks=n_test_tasks)


        runner.train(n_epochs=n_epochs,
                     batch_size=episode_per_task * max_path_length *
                     meta_batch_size)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
