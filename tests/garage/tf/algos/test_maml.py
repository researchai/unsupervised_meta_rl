import tensorflow as tf

from garage.envs.point_env import PointEnv
from garage.envs.multitask_env import MultitaskEnv
from garage.misc.instrument import run_experiment
from garage.tf.algos.maml import MAML
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.policies.maml_policy import MamlPolicy

from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDiscreteEnv


def test_maml_policy(*_):
    tasks = [(-3., 0.), (3., 0.), (0., 3.), (0., -3.)]
    env = TfEnv(MultitaskEnv(PointEnv(), tasks))
    policy = GaussianMLPPolicy(
            env_spec=env.spec, hidden_sizes=(32, 32))
    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(hidden_sizes=(32, 32)),
    )
    maml_policy = MamlPolicy(wrapped_policy=policy, n_tasks=4)
    
    algo = MAML(
        policy=maml_policy,
        baseline=baseline,
        env=env,
        max_path_length=100,
    )
    algo.train()

    # gradient_vars = list()
    # params = policy.get_params_internal()

    # for i in range(2):
    #     g_i = list()
    #     for p in params:
    #         grad = tf.placeholder(
    #             dtype=p.dtype, 
    #             shape=p.shape, 
    #             name="maml/{}/grad/{}".format(i, p.name[:-2])
    #         )
    #         g_i.append(grad)
    #     gradient_vars.append(g_i)

    # maml_policy.initialize(gradient_var=gradient_vars)
    # sess = tf.Session()
    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("./graphh", sess.graph)

run_experiment(
    test_maml_policy,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)