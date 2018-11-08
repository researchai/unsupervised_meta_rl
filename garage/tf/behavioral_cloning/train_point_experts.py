import numpy as np
import tensorflow as tf

from garage.envs import normalize
from garage.misc.instrument import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.behavioral_cloning.point_env import PointEnv
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


def circle(r, n):
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
        yield r * np.sin(t), r * np.cos(t)


N = 4
goals = circle(3.0, N)
TASKS = {
    str(i + 1): {
        'args': [],
        'kwargs': {
            'goal': g
        }
    }
    for i, g in enumerate(goals)
}

for i, task in TASKS.items():

    def run_task(*_):
        with tf.Graph().as_default():
            env = TfEnv(normalize(PointEnv(**task["kwargs"])))

            policy = GaussianMLPPolicy(
                name="policy",
                env_spec=env.spec,
                hidden_sizes=(20, 20),
                hidden_nonlinearity=tf.nn.relu)

            baseline = GaussianMLPBaseline(env_spec=env.spec)

            algo = PPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=4096,
                max_path_length=50,
                n_itr=500,
                discount=0.99,
                step_size=0.2,
                optimizer_args=dict(batch_size=32, max_epochs=10),
                plot=True)

            with tf.Session() as sess:
                algo.train(sess)
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    w0 = tf.get_variable("policy/mean_network/hidden_0/W")
                    b0 = tf.get_variable("policy/mean_network/hidden_0/b")
                    w1 = tf.get_variable("policy/mean_network/hidden_1/W")
                    b1 = tf.get_variable("policy/mean_network/hidden_1/b")
                    wout = tf.get_variable("policy/mean_network/output/W")
                    bout = tf.get_variable("policy/mean_network/output/b")
                    param = tf.get_variable(
                        "policy/std_network/output_std_param/param")
                    var_to_save = [w0, b0, w1, b1, wout, bout, param]
                    saver = tf.train.Saver({v.op.name: v for v in var_to_save})
                    saver.save(
                        sess,
                        "./ppo_point_expert/ppo_task_" + str(i) + ".ckpt")

    run_experiment(
        run_task, n_parallel=12, exp_prefix="ppo_point", seed=1, plot=True)
