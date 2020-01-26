#!/usr/bin/env python3
"""
This is an example to train a task with PPO algorithm.
Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.
Results:
    AverageDiscountedReturn: 500
    RiseTime: itr 40
"""
import tensorflow as tf
from garage.envs.multi_env_wrapper import MultiEnvWrapper
from garage.experiment import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
from metaworld.envs.mujoco.env_dict import EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT
from garage.envs import GarageEnv

MT10_envs_by_id = {}

for (task, env) in EASY_MODE_CLS_DICT.items():
    MT10_envs_by_id[task] = GarageEnv(env(*EASY_MODE_ARGS_KWARGS[task]['args'],
                                **EASY_MODE_ARGS_KWARGS[task]['kwargs']))

MT10_env_ids = ['reach-v1',
                'push-v1',
                'pick-place-v1',
                'door-v1',
                'drawer-open-v1',
                'drawer-close-v1',
                'button-press-topdown-v1',
                'ped-insert-side-v1',
                'window-open-v1',
                'window-close-v1']
MT10_envs = [MT10_envs_by_id[i] for i in MT10_env_ids]
hyper_parameters = {
    'hidden_sizes': [64, 64],
    'center_adv': True,
    'learning_rate': 1e-3,
    'lr_clip_range': 0.2,
    'gae_lambda': 0.95,
    'discount': 0.99,
    'policy_ent_coeff': 0.0,
    'max_path_length': 100,
    'batch_size': 2048,
    'n_epochs': 10,
    'n_trials': 1,
    'training_batch_size': 32,
    'training_epochs': 4,
}
def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = MultiEnvWrapper(MT10_envs)
        import ipdb; ipdb.set_trace()
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(32, 32),
                use_trust_region=True,
            ),
        )
        # NOTE: make sure when setting entropy_method to 'max', set
        # center_adv to False and turn off policy gradient. See
        # tf.algos.NPO for detailed documentation.
        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
            ),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=0.02,
            center_adv=False,
        )
        runner.setup(algo, env)
        runner.train(n_epochs=120, batch_size=2048, plot=False)
run_experiment(run_task, snapshot_mode='last', seed=1)