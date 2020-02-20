import sys

from types import SimpleNamespace

from akro.tf import Box

from garage.envs.env_spec import EnvSpec
from garage.experiment import run_experiment
import numpy as np

from embed2learn.algos import PPOTaskEmbedding
from embed2learn.baselines import MultiTaskGaussianMLPBaseline, MultiTaskLinearFeatureBaseline
from embed2learn.envs import MultiTaskEnv
from embed2learn.envs.multi_task_env import TfEnv
from embed2learn.embeddings import EmbeddingSpec
from embed2learn.embeddings import GaussianMLPEmbedding
from embed2learn.embeddings.utils import concat_spaces
from embed2learn.experiment import TaskEmbeddingRunner
from embed2learn.policies import GaussianMLPMultitaskPolicy


from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv


N_TASKS = 50
SEED = sys.argv[1]


def run_task(v):

    from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, HARD_MODE_ARGS_KWARGS
    MT50_CLS_DICT = {}
    MT50_ARGS_KWARGS = {}
    for k in HARD_MODE_CLS_DICT.keys():
        for i in HARD_MODE_CLS_DICT[k].keys():
            key = '{}-{}'.format(k, i)
            MT50_CLS_DICT[key] = HARD_MODE_CLS_DICT[k][i]
            MT50_ARGS_KWARGS[key] = HARD_MODE_ARGS_KWARGS[k][i]

    assert len(MT50_CLS_DICT.keys()) == N_TASKS
    mt50 = MultiClassMultiTaskEnv(
                task_env_cls_dict=MT50_CLS_DICT,
                task_args_kwargs=MT50_ARGS_KWARGS,
                sample_goals=False,
                obs_type='plain',
                sample_all=True,)

    goals_dict = {
        t: [e.goal.copy()]
        for t, e in zip(mt50._task_names, mt50._task_envs)}
    mt50.discretize_goal_space(goals_dict)
    # reach, push, pickplace are different
    mt50._task_envs[0].task_type = 'reach'
    mt50._task_envs[1].task_type = 'pickplace'
    mt50._task_envs[2].task_type = 'push'
    mt50._task_envs[3].task_type = 'reach'
    mt50._task_envs[4].task_type = 'pickplace'
    mt50._task_envs[5].task_type = 'push'
    mt50._task_envs[0].goal = np.array([-0.1, 0.8, 0.2])
    mt50._task_envs[1].goal = np.array([0.1, 0.8, 0.2])
    mt50._task_envs[2].goal = np.array([0.1, 0.8, 0.02])
    mt50._task_envs[3].goal = np.array([-0.05, 0.8, 0.2])
    mt50._task_envs[4].goal = np.array([0.05, 0.8, 0.2])
    mt50._task_envs[5].goal = np.array([0.05, 0.8, 0.015])

    env = mt50

    with TaskEmbeddingRunner() as runner:
        v = SimpleNamespace(**v)

        # Environment
        env = TfEnv(mt50)

        # Latent space and embedding specs
        # TODO(gh/10): this should probably be done in Embedding or Algo
        latent_lb = np.zeros(v.latent_length, )
        latent_ub = np.ones(v.latent_length, )
        latent_space = Box(latent_lb, latent_ub)

        # trajectory space is (TRAJ_ENC_WINDOW, act_obs) where act_obs is a stacked
        # vector of flattened actions and observations
        act_lb, act_ub = env.action_space.bounds
        act_lb_flat = env.action_space.flatten(act_lb)
        act_ub_flat = env.action_space.flatten(act_ub)
        obs_lb, obs_ub = env.observation_space.bounds
        obs_lb_flat = env.observation_space.flatten(obs_lb)
        obs_ub_flat = env.observation_space.flatten(obs_ub)
        # act_obs_lb = np.concatenate([act_lb_flat, obs_lb_flat])
        # act_obs_ub = np.concatenate([act_ub_flat, obs_ub_flat])
        act_obs_lb = obs_lb_flat
        act_obs_ub = obs_ub_flat
        # act_obs_lb = act_lb_flat
        # act_obs_ub = act_ub_flat
        traj_lb = np.stack([act_obs_lb] * v.inference_window)
        traj_ub = np.stack([act_obs_ub] * v.inference_window)
        traj_space = Box(traj_lb, traj_ub)

        task_embed_spec = EmbeddingSpec(env.task_space, latent_space)
        traj_embed_spec = EmbeddingSpec(traj_space, latent_space)
        task_obs_space = concat_spaces(env.task_space, env.observation_space)
        env_spec_embed = EnvSpec(task_obs_space, env.action_space)

        # TODO(): rename to inference_network
        traj_embedding = GaussianMLPEmbedding(
            name="inference",
            embedding_spec=traj_embed_spec,
            hidden_sizes=(200, 100),
            std_share_network=True,
            init_std=2.0,
        )

        # Embeddings
        task_embedding = GaussianMLPEmbedding(
            name="embedding",
            embedding_spec=task_embed_spec,
            hidden_sizes=(200, 200),
            std_share_network=True,
            init_std=v.embedding_init_std,
            max_std=v.embedding_max_std,
        )

        # Multitask policy
        policy = GaussianMLPMultitaskPolicy(
            name="policy",
            env_spec=env.spec,
            task_space=env.task_space,
            embedding=task_embedding,
            hidden_sizes=(400, 400, 400),
            std_share_network=True,
            init_std=v.policy_init_std,
            max_std=5.,
        )

        extra = v.latent_length + N_TASKS
        # baseline = MultiTaskGaussianMLPBaseline(
        #     env_spec=env.spec,
        #     extra_dims=extra,
        #     regressor_args=dict(hidden_sizes=(200, 100)),
        # )
        baseline = MultiTaskLinearFeatureBaseline(env_spec=env.spec,)

        algo = PPOTaskEmbedding(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            inference=traj_embedding,
            max_path_length=v.max_path_length,
            n_itr=20000,
            discount=0.99,
            lr_clip_range=0.2,
            policy_ent_coeff=v.policy_ent_coeff,
            embedding_ent_coeff=v.embedding_ent_coeff,
            inference_ce_coeff=v.inference_ce_coeff,
            use_softplus_entropy=True,
        )
        runner.setup(algo, env, batch_size=v.batch_size,
            max_path_length=v.max_path_length)
        runner.train(n_epochs=200000, plot=False)

config = dict(
    latent_length=10,
    inference_window=20,
    batch_size=20 * N_TASKS * 150,
    policy_ent_coeff=1e-3,  # 1e-2
    embedding_ent_coeff=1e-4,  # 1e-3
    inference_ce_coeff=5e-3,  # 1e-4
    max_path_length=150,
    embedding_init_std=1.0,
    embedding_max_std=2.0,
    policy_init_std=2.0,
)

run_experiment(
    run_task,
    python_command='python -W ignore',
    exp_prefix='mt50_seed{}'.format(SEED),
    n_parallel=4,
    seed=SEED,
    variant=config,
    snapshot_mode='last',
    plot=False,
)
