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

        # retrieve paramters before running re-init
        policy_params_before = policy.get_param_values()
        baseline_params_before = baseline.get_param_values()

        # This line below will re-init everything...
        # Still waiting for the garage team to resolve this problem.
        # See https://github.com/rlworkgroup/garage/issues/511 for
        # details.
        sess.run(tf.global_variables_initializer())

        # Setting params values to the unpacked ones
        policy.set_param_values(policy_params_before)
        baseline.set_param_values(baseline_params_before)

        # This is kinda messy now.
        # The adaptation step have to be done with a single
        # task sampler. 
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

    env = TfEnv(PointEnv(goal=(2, 2)))
    adapt_policy(sys.argv[1], env)
