import copy
from garage.envs import RL2Env
from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.envs.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.experiment import run_experiment
from garage.experiment import task_sampler
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import RL2
from garage.tf.algos import RL2PPO
from garage.tf.algos import RL2TRPO
from garage.tf.algos import RL2PPO2
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import GaussianGRUPolicy
from garage.sampler.rl2_sampler import RL2Sampler
from garage.sampler import LocalSampler
from garage.sampler import RaySampler
from garage.sampler.rl2_worker import RL2Worker

from metaworld.benchmarks import ML1
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
        meta_batch_size = 10
        n_epochs = 10
        episode_per_task = 2
        steps_per_epoch = 1
        n_test_tasks = 1

        # ---- For ML1-push
        # env = RL2Env(ML1.get_train_tasks('push-v1'))
        # tasks = task_sampler.EnvPoolSampler([env])
        # tasks.grow_pool(meta_batch_size)

        # ---- For HalfCheetahVel
        tasks = task_sampler.SetTaskSampler(lambda: RL2Env(env=HalfCheetahVelEnv()))
        
        env_spec = tasks.sample(1)[0]().spec
        policy = GaussianGRUPolicy(name='policy',
                                    hidden_dims=[64, 64],
                                    env_spec=env_spec,
                                    state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        inner_algo = RL2TRPO(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         max_path_length=max_path_length * episode_per_task,
                         discount=0.99,
                         max_kl_step=0.01,
                         optimizer=ConjugateGradientOptimizer,
                         gae_lambda=0.95,
                         optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                         base_eps=1e-5)))

        algo = RL2(policy=policy,
                   inner_algo=inner_algo,
                   max_path_length=max_path_length,
                   meta_batch_size=meta_batch_size,
                   task_sampler=tasks,
                   steps_per_epoch=steps_per_epoch)

        runner.setup(algo,
                     tasks.sample(meta_batch_size),
                     sampler_cls=LocalSampler,
                     n_workers=meta_batch_size,
                     worker_class=RL2Worker)

        # runner.setup_meta_evaluator(test_task_sampler=tasks,
        #                             sampler_cls=LocalSampler,
        #                             n_test_tasks=n_test_tasks)

        runner.train(n_epochs=n_epochs,
                     batch_size=episode_per_task * max_path_length *
                     meta_batch_size)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
