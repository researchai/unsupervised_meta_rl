import sys

import joblib
import tensorflow as tf

from garage.envs.point_env import PointEnv
from garage.tf.algos import MAML
from garage.tf.samplers import OnPolicyVectorizedSampler


def adapt_policy(pkl_path, env):
    with tf.Session():
        snapshot = joblib.load(pkl_path)

        policy = snapshot['policy']
        baseline = snapshot['baseline']

        algo = MAML(
            policy=policy,
            baseline=baseline,
            env=env,
            sampler_cls=OnPolicyVectorizedSampler,
            max_path_length=100,
        )

        params = algo.adapt_policy()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s PKL_FILENAME' % sys.argv[0])
        sys.exit(0)

    env = PointEnv()
    adapt_policy(sys.argv[1], env)
