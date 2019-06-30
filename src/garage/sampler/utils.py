import signal
import time

import numpy as np

from garage.misc import tensor_utils

from garage.experiment import snapshotter
def mt_rollout(snapshot_dir, 
            load_itr='all',
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False):
    snapshotter.snapshot_dir = snapshot_dir
    saved = snapshotter.load(load_itr)
    # print(saved)
    envs = saved['env']
    agent = saved['algo'].policy
    n_tasks = len(envs)
    task_one_hots = np.eye(n_tasks)[list(range(n_tasks))]

    samples_data_list = []
    # print(envs)
    for i, (env, t) in enumerate(zip(envs, task_one_hots)):
        observations = []
        # tasks = []
        # latents = []
        # latent_infos = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        o = env.reset()
        if animated:
            env.render()

        path_length = 0
        while path_length < max_path_length:
            a, agent_info = agent.get_action(np.concatenate((o, t)))
            # a, agent_info = agent.get_action_from_latent(z, o)
            # latent_info = agent_info["latent_info"]
            next_o, r, d, env_info = env.step(a)
            observations.append(agent.observation_space.flatten(o))
            # tasks.append(t)
            # z = latent_info["mean"]
            # latents.append(agent.latent_space.flatten(z))
            # latent_infos.append(latent_info)
            rewards.append(r)
            actions.append(agent.action_space.flatten(a))
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o
            if animated:
                env.render()
                timestep = 0.05
                time.sleep(timestep / speedup)

        samples_data = dict(
            observations=tensor_utils.stack_tensor_list(observations),
            actions=tensor_utils.stack_tensor_list(actions),
            rewards=tensor_utils.stack_tensor_list(rewards),
            # tasks=tensor_utils.stack_tensor_list(tasks),
            # latents=tensor_utils.stack_tensor_list(latents),
            # latent_infos=tensor_utils.stack_tensor_dict_list(latent_infos),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        )
        samples_data_list.append(samples_data)
    
    if animated and not always_return_paths:
        return
    
    return samples_data_list

def rollout(env,
            agent,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return None

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )


def truncate_paths(paths, max_samples):
    """
    Truncate the list of paths so that the total number of samples is exactly
    equal to max_samples. This is done by removing extra paths at the end of
    the list, and make the last path shorter if necessary
    :param paths: a list of paths
    :param max_samples: the absolute maximum number of samples
    :return: a list of paths, truncated so that the number of samples adds up
    to max-samples
    """
    # chop samples collected by extra paths
    # make a copy
    paths = list(paths)
    total_n_samples = sum(len(path['rewards']) for path in paths)
    while paths and total_n_samples - len(paths[-1]['rewards']) >= max_samples:
        total_n_samples -= len(paths.pop(-1)['rewards'])
    if paths:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(
            last_path['rewards']) - (total_n_samples - max_samples)
        for k, v in last_path.items():
            if k in ['observations', 'actions', 'rewards']:
                truncated_last_path[k] = tensor_utils.truncate_tensor_list(
                    v, truncated_len)
            elif k in ['env_infos', 'agent_infos']:
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(
                    v, truncated_len)
            else:
                raise NotImplementedError
        paths.append(truncated_last_path)
    return paths


def center_advantages(advantages):
    return (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)


def shift_advantages_to_positive(advantages):
    return (advantages - np.min(advantages)) + 1e-8


def sign(x):
    return 1. * (x >= 0) - 1. * (x < 0)


class MaskSignals():
    """Context Manager to mask a list of signals."""

    def __init__(self, signals):
        self.signals = signals

    def __enter__(self):
        signal.pthread_sigmask(signal.SIG_BLOCK, self.signals)

    def __exit__(self, *args):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, self.signals)


mask_signals = MaskSignals
