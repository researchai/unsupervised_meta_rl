# from garage.envs import HalfCheetahVelEnv
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from garage.envs import RL2Env
from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO
from garage.tf.algos import RL2
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import GaussianGRUPolicy
from garage.sampler.rl2_sampler import RL2Sampler

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
        # env = RL2Env(env=HalfCheetahRandDirecEnv())
        # env = RL2Env(ML1.get_train_tasks('pick-place-v1'))
        # policy = GaussianGRUPolicy(name='policy', hidden_dim=64, env_spec=env.spec, state_include_action=False)

        # baseline = LinearFeatureBaseline(env_spec=env.spec)

        # max_path_length = 100
        # meta_batch_size = 200
        # n_epochs = 500
        # episode_per_task = 10

        # inner_algo = PPO(env_spec=env.spec,
        #                  policy=policy,
        #                  baseline=baseline,
        #                  max_path_length=max_path_length * episode_per_task,
        #                  discount=0.99,
        #                  lr_clip_range=0.2,
        #                  meta_learn=True,
        #                  num_of_env=meta_batch_size,
        #                  episode_per_task=episode_per_task,
        #                  optimizer_args=dict(max_epochs=5))

        # runner.setup(algo, env)
        # runner.train(n_epochs=n_epochs, batch_size=200000)
        runner.restore("./")
        runner.resume(n_epochs=1000, batch_size=None)

run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
