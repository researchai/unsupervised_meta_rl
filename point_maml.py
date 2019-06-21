import numpy as np
import tensorflow as tf

from garage.envs.point_env import PointEnv
from garage.experiment import run_experiment
from garage.experiment.local_tf_maml_runner import LocalMamlRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos.maml import MAML
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies.gaussian_mlp_policy_with_model import GaussianMLPPolicyWithModel as GaussianMLPPolicy
from garage.tf.policies.maml_policy import MamlPolicy
from garage.tf.optimizers import ConjugateGradientOptimizer

from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDiscreteEnv


def circle(r, n):
    tasks = list()
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
        tasks.append((r * np.sin(t), r * np.cos(t)))
    return tasks


def test_maml(*_):
    with LocalMamlRunner() as runner:
        # tasks = [
            # (-3., 0.),
            # (3., 0.),
            # (0., 3.),
            # (0., -3.),
            # (-1.5, 1.5),
            # (1.5, -1.5),
            # (1.5, 1.5),
            # (-1.5, -1.5),
        # ]
        # tasks = circle(3., 8)
        num_goals = 2000
        tasks = np.random.uniform(-3., 3., size=(num_goals, 2, )).tolist()
        envs = [TfEnv(PointEnv(goal=np.array(t))) for t in tasks]

        n_tasks = len(envs)
        policy = GaussianMLPPolicy(
                env_spec=envs[0].spec,
                hidden_nonlinearity=tf.nn.relu,
                hidden_sizes=(100, 100),)
        baseline = LinearFeatureBaseline(
            env_spec=envs[0].spec,
            #regressor_args=dict(hidden_sizes=(100, 100)),
        )
        maml_policy = MamlPolicy(wrapped_policy=policy, meta_batch_size=20)
        algo = MAML(
            env_spec=envs[0].spec,
            policy=maml_policy,
            baseline=baseline,
            env=envs,
            optimizer=ConjugateGradientOptimizer,
            max_path_length=50,
        )
        runner.setup(algo, envs)
        runner.train(n_epochs=200, batch_size=20*2000)

run_experiment(
    test_maml,
    n_parallel=2,
    exp_prefix='maml_8goals',
    seed=1,
    plot=False,
)
