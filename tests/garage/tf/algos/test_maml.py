import tensorflow as tf

from garage.envs.point_env import PointEnv
from garage.envs.multitask_env import MultitaskEnv
from garage.misc.instrument import run_experiment
from garage.tf.algos.maml import MAML
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.policies.maml_policy import MamlPolicy
from garage.tf.optimizers import ConjugateGradientOptimizer

from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDiscreteEnv


def test_maml_policy(*_):
    tasks = [(-3., 0.), (3., 0.), (0., 3.), (0., -3.)]
    env = TfEnv(MultitaskEnv(PointEnv(), tasks))
    policy = GaussianMLPPolicy(
            env_spec=env.spec, hidden_sizes=(100, 100))
    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(hidden_sizes=(100, 100)),
    )
    maml_policy = MamlPolicy(wrapped_policy=policy, n_tasks=4)
    
    algo = MAML(
        policy=maml_policy,
        baseline=baseline,
        env=env,
        optimizer=ConjugateGradientOptimizer,
        max_path_length=100,
    )
    algo.train()


run_experiment(
    test_maml_policy,
    n_parallel=1,
    exp_prefix='maml',
    seed=1,
    plot=False,
)
