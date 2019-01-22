import tensorflow as tf

from garage.tf.algos.maml import MAML
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.policies.maml_policy import MamlPolicy

from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDiscreteEnv


def test_maml_policy():
    box_env = TfEnv(DummyBoxEnv())
    policy = GaussianMLPPolicy(
            env_spec=box_env, hidden_sizes=(1, ))

    maml_policy = MamlPolicy(wrapped_policy=policy, n_tasks=2)
    
    algo = MAML(policy=maml_policy)

    gradient_vars = list()
    params = policy.get_params_internal()

    for i in range(2):
        g_i = list()
        for p in params:
            grad = tf.placeholder(
                dtype=p.dtype, 
                shape=p.shape, 
                name="maml/{}/grad/{}".format(i, p.name[:-2])
            )
            g_i.append(grad)
        gradient_vars.append(g_i)

    maml_policy.initialize(gradient_var=gradient_vars)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./graphh", sess.graph)

test_maml_policy()