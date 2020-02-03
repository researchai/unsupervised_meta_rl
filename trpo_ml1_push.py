#!/usr/bin/env python3
"""
This is an example to train a task with PPO algorithm.

Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.

Results:
    AverageDiscountedReturn: 500
    RiseTime: itr 40

"""
import random

from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.envs.ml1_wrapper import ML1WithPinnedGoal
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment.deterministic import set_seed
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
from metaworld.envs.mujoco.env_dict import EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT

MT10_envs_by_id = {
    task: env(*EASY_MODE_ARGS_KWARGS[task]['args'],
              **EASY_MODE_ARGS_KWARGS[task]['kwargs'])
    for (task, env) in EASY_MODE_CLS_DICT.items()
}

@wrap_experiment
def trpo_ml1_pick_place(ctxt=None, seed=1):

    env_id = "push-v1"

    """Run task."""
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        ret_envs, ret_names = get_ML1_envs_test(env_id)
        env = MultiEnvWrapper(ret_envs, ret_names, sample_strategy=round_robin_strategy)

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(64, 64))

        # baseline = LinearFeatureBaseline(env_spec=env.spec)
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=False,
            ),
        )

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=150,
                    discount=0.99,
                    gae_lambda=0.97,
                    max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=999999, batch_size=4096, plot=False)


def get_ML1_envs_test(name):
    bench = ML1WithPinnedGoal.get_train_tasks(name)
    tasks = [{'task': 0, 'goal': i} for i in range(50)]
    ret = dict()
    ret_envs = []
    ret_names = []
    for task in tasks:
        new_bench = bench.clone(bench)
        new_bench.set_task(task)
        ret_names.append(("goal"+str(task['goal'])))
        ret_envs.append(GarageEnv((new_bench.active_env)))
    return ret_envs, ret_names


seeds = random.sample(range(100), 1)
for seed in seeds:
    trpo_ml1_pick_place(seed=seed)
