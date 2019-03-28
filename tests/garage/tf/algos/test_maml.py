import numpy as np
import tensorflow as tf

from garage.envs.point_env import PointEnv
from garage.experiment import run_experiment
from garage.experiment.local_tf_maml_runner import LocalMamlRunner
from garage.tf.algos.maml import MAML
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies.gaussian_mlp_policy_with_model import GaussianMLPPolicyWithModel as GaussianMLPPolicy
from garage.tf.policies.maml_policy import MamlPolicy
from garage.tf.optimizers import ConjugateGradientOptimizer

from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDiscreteEnv


def test_maml(*_):
    with LocalMamlRunner() as runner:
        tasks = [(-3., 0.), (3., 0.), (0., 3.), (0., -3.)]
        envs = [TfEnv(PointEnv(goal=np.array(t))) for t in tasks]
        policy = GaussianMLPPolicy(
                env_spec=envs[0].spec, hidden_sizes=(100, 100))
        baseline = GaussianMLPBaseline(
            env_spec=envs[0].spec,
            regressor_args=dict(hidden_sizes=(100, 100)),
        )
        maml_policy = MamlPolicy(wrapped_policy=policy, n_tasks=4)
        
        algo = MAML(
            policy=maml_policy,
            baseline=baseline,
            env=envs,
            optimizer=ConjugateGradientOptimizer,
            max_path_length=100,
        )

        runner.setup(algo, envs)
        runner.train(n_epochs=2, batch_size=4*1000)

run_experiment(
    test_maml,
    n_parallel=1,
    exp_prefix='maml',
    seed=1,
    plot=False,
)

