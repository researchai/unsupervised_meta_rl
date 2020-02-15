#!/usr/bin/env python3
"""
This is an example to train a task with PPO algorithm.

Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.

Results:
    AverageDiscountedReturn: 500
    RiseTime: itr 40

"""
import sys

import tensorflow as tf
from garage import wrap_experiment
from garage.envs import GarageEnv, normalize_reward
from garage.envs.ml1_wrapper import ML1WithPinnedGoal
from garage.envs.multi_task_metaworld_wrapper import MTMetaWorldWrapper
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy

# env_id = 'push-v1'
# env_id = 'reach-v1'
env_id = 'pick-place-v1'

@wrap_experiment
def ppo_ml1(ctxt=None, seed=1):

    """Run task."""
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        Ml1_reach_envs = get_ML1_envs_test(env_id)
        env = MTMetaWorldWrapper(Ml1_reach_envs)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        # baseline = LinearFeatureBaseline(env_spec=env.spec)
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=False,
            ),
        )

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=150,
            discount=0.99,
            gae_lambda=0.97,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                tf_optimizer_args=dict(
                    learning_rate=3e-4,
                ),
            ),
        )

        timesteps = 20000000
        batch_size = 150 * env.num_tasks
        epochs = timesteps // batch_size

        print (f'epochs: {epochs}, batch_size: {batch_size}')

        runner.setup(algo, env)
        runner.train(n_epochs=epochs, batch_size=batch_size, plot=False)


def get_ML1_envs_test(name):
    bench = ML1WithPinnedGoal.get_train_tasks(name)
    tasks = [{'task': 0, 'goal': i} for i in range(50)]
    ret = {}
    import pdb; pdb.set_trace()
    for task in tasks:
        new_bench = bench.clone(bench)
        new_bench.set_task(task)
        ret[("goal"+str(task['goal']))] = GarageEnv(normalize_reward(new_bench.active_env))
    return ret


s = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ppo_ml1(seed=s)
