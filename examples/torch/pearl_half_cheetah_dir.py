import akro
import numpy as np

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.envs.env_spec import EnvSpec
from garage.envs.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.experiment import LocalRunner, run_experiment
from garage.sampler import PEARLSampler
from garage.torch.algos import PEARLSAC
from garage.torch.embeddings import RecurrentEncoder
from garage.torch.modules import MLPEncoder
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.policies import ContextConditionedPolicy, \
    TanhGaussianMLPPolicy
import garage.torch.utils as tu

params = dict(
    num_epochs=50,
    num_train_tasks=2,
    num_eval_tasks=2,
    latent_size=5, # dimension of the latent context vector
    net_size=300, # number of units per FC layer in each network
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
    ),
    algo_params=dict(
        meta_batch=4, # number of tasks to average the gradient across
        num_steps_per_epoch=2000, # number of meta-gradient steps taken per iteration
        num_initial_steps=2000, # number of transitions collected per task before training
        num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=1000, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=1000, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_evals=2, # number of independent evals
        num_steps_per_eval=600,  # nuumber of transitions to eval on
        batch_size=256, # number of transitions in the RL batch
        embedding_batch_size=256, # number of transitions in the context batch
        embedding_mini_batch_size=256, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3e-4,
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=2, # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False, # recurrent or permutation-invariant encoder
        use_information_bottleneck=True, # False makes latent context deterministic
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
    ),
)


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        _ : Unused parameters

    """
    # create multi-task environment and sample tasks
    env = GarageEnv(normalize(HalfCheetahDirEnv()))
    runner = LocalRunner(snapshot_config)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = params['latent_size']
    encoder_in_dim = 2 * obs_dim + action_dim + reward_dim \
        if params['algo_params']['use_next_obs_in_context'] \
            else obs_dim + action_dim + reward_dim
    encoder_out_dim = latent_dim * 2 if params['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = params['net_size']
    recurrent = params['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MLPEncoder

    context_encoder = encoder_model(input_dim=encoder_in_dim,
                                    output_dim=encoder_out_dim,
                                    hidden_sizes=[200, 200, 200])

    space_a = akro.Box(low=-1, high=1, shape=(obs_dim+latent_dim, ), dtype=np.float32)
    space_b = akro.Box(low=-1, high=1, shape=(action_dim, ), dtype=np.float32)
    qf_env = EnvSpec(space_a, space_b)

    qf1 = ContinuousMLPQFunction(env_spec=qf_env,
                                 hidden_sizes=[net_size, net_size, net_size])

    qf2 = ContinuousMLPQFunction(env_spec=qf_env,
                                 hidden_sizes=[net_size, net_size, net_size])

    obs_space = akro.Box(low=-1, high=1, shape=(obs_dim, ), dtype=np.float32)
    action_space = akro.Box(low=-1, high=1, shape=(latent_dim, ), dtype=np.float32)
    vf_env = EnvSpec(obs_space, action_space)

    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    policy = TanhGaussianMLPPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    
    agent = ContextConditionedPolicy(
        latent_dim=latent_dim,
        context_encoder=context_encoder,
        policy=policy,
        use_ib=params['algo_params']['use_information_bottleneck'],
        use_next_obs=params['algo_params']['use_next_obs_in_context'],
    )

    pearlsac = PEARLSAC(
        env=env,
        num_train_tasks=params['num_train_tasks'],
        num_eval_tasks=params['num_eval_tasks'],
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **params['algo_params']
    )

    tu.set_gpu_mode(False)
    #pearlsac.to()

    runner.setup(algo=pearlsac, env=env, sampler_cls=PEARLSampler,
        sampler_args=dict(max_path_length=params['algo_params']['max_path_length']))
    runner.train(n_epochs=params['num_epochs'], batch_size=256)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
