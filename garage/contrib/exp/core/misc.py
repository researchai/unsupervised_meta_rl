from collections import namedtuple

import numpy as np


def _get_flat_dim(space):
    return np.prod(space.low.shape)

def get_env_spec(env):
    Spec = namedtuple('EnvSpec', ['action_space', 'observation_space'])
    Space = namedtuple('Space', 'flat_dim')
    action_space = env.action_space
    obs_space = env.observation_space
    spec = Spec(Space(_get_flat_dim(action_space)), Space(_get_flat_dim(obs_space)))
    return spec
