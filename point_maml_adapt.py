import argparse

import joblib
import numpy as np
import tensorflow as tf

from garage.experiment import run_experiment
from garage.experiment.local_tf_maml_runner import LocalMamlRunner
from garage.tf.algos.maml import MAML
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.policies.maml_policy import MamlPolicy
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.envs.dm_control import DmControlEnv
from garage.tf.samplers import OnPolicyVectorizedSampler
from garage.envs.point_env import PointEnv


def evaluate_once(policy,
                  env,
                  max_path_length=100,):
    observations = []
    actions = []
    rewards = []
    action_infos = []

    o = env.reset()
    for i in range(max_path_length):
        env.render()
        observations.append(o)
        a, info = policy.get_actions(np.array([o]))
        o, r, d, _ = env.step(a[0])

        actions.append(a)
        rewards.append(r)
        action_infos.append(info) 
        if d or i == (max_path_length - 1):
            observations.append(o)
            break

    results = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        action_infos=action_infos,
    )
    return results


def adapt_policy_and_evaluate(pkl_path, envs, eval_epoch=1):
    with LocalMamlRunner() as runner:

        snapshot = joblib.load(pkl_path)
        policy = snapshot['policy']
        baseline = snapshot['baseline']

        # retrieve paramters before running re-init
        sess = runner.sess
        policy_params_before = sess.run(policy.get_params())
        baseline_params_before = baseline.get_param_values()
        maml_policy = MamlPolicy(wrapped_policy=policy, n_tasks=8)

        # This line below will re-init everything...
        # Still waiting for the garage team to resolve this problem.
        # See https://github.com/rlworkgroup/garage/issues/511 for
        # details.
        # sess.run(tf.global_variables_initializer())

        # This is kinda messy now.
        # The adaptation step have to be done with a single
        # task sampler. 
        algo = MAML(
            env_spec=envs[0].spec,
            policy=maml_policy,
            baseline=baseline,
            env=envs[0],
            sampler_cls=OnPolicyVectorizedSampler,
            max_path_length=100,
        )
        runner.setup(algo, envs)

        results = list()
        for env in envs:
            # Setting params values to the unpacked ones
            maml_policy.update_params(policy_params_before)
            baseline.set_param_values(baseline_params_before)
            # Retrieve adapted params
            params = runner.adapt(env=env)
            maml_policy.update_params(params)
            task_result = []
            for _ in range(eval_epoch):
                ep_result = evaluate_once(maml_policy, env)
                ep_result['goal'] = env.env._goal
                task_result.append(ep_result)
                results.append(task_result)

        rewards = [re['rewards'] for re in task_result]
        returns = [np.sum(re) for re in rewards]
        returns_mean = np.mean(returns)

    return returns_mean, results

def circle(r, n):
    tasks = list()
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
        tasks.append((r * np.sin(t), r * np.cos(t)))
    return tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'variation',
    #     help='Variation of a mdp for meta learning.',
    #     type=str,
    #     choices=VARIATIONS.keys())
    parser.add_argument(
        'pkl_path',
        help='Path of the saved meta-learned policy',
        type=str,)
    parser.add_argument(
        '--n_parallel',
        help='Number of process to sample from environments',
        type=int,
        default=1)
    parser.add_argument(
        '--test_set_size',
        help='Number of test environments',
        type=int,
        default=1)
    args = parser.parse_args()

    test_tasks = [(0, 3)]
    test_envs = [TfEnv(PointEnv(goal=np.array(t), never_done=False)) for t in test_tasks]
    
    returns_mean, results = adapt_policy_and_evaluate(
        pkl_path=args.pkl_path,
        envs=test_envs,
        eval_epoch=10,
    )

    np.save('adaptation_data_16.npy', np.array(results))
    import ipdb
    ipdb.set_trace()
    print('Exiting')