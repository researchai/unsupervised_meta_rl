import sys

import joblib
import tensorflow as tf

from garage.envs.point_env import PointEnv
from garage.tf.envs import TfEnv
from garage.tf.algos import MAML
from garage.tf.samplers import OnPolicyVectorizedSampler


def adapt_policy(pkl_path, env):
    sess = tf.Session()
    with sess.as_default():

        snapshot = joblib.load(pkl_path)
        policy = snapshot['policy']
        baseline = snapshot['baseline']
        p_before = sess.run(baseline.get_params_internal())
        sess.run(tf.global_variables_initializer())
        p_after = sess.run(baseline.get_params_internal())

        algo = MAML(
            policy=policy,
            baseline=baseline,
            env=env,
            sampler_cls=OnPolicyVectorizedSampler,
            max_path_length=100,
        )

        params = algo.adapt_policy(sess=sess)
        policy.update_params(params)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s PKL_FILENAME' % sys.argv[0])
        sys.exit(0)

    env = TfEnv(PointEnv())
    adapt_policy(sys.argv[1], env)
