"""Garage wrappers for gym environments."""

from garage.envs.base import GarageEnv
from garage.envs.base import Step
from garage.envs.env_spec import EnvSpec
from garage.envs.grid_world_env import GridWorldEnv
from garage.envs.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.envs.ml1_wrapper import ML1WithPinnedGoal
from garage.envs.normalized_env import normalize
from garage.envs.normalized_reward_env import normalize_reward
from garage.envs.point_env import PointEnv
from garage.envs.rl2_env import RL2Env
from garage.envs.task_id_wrapper import TaskIdWrapper
from garage.envs.task_id_wrapper2 import TaskIdWrapper2
from garage.envs.ignore_done_wrapper import IgnoreDoneWrapper
from garage.envs.task_onehot_wrapper import TaskOnehotWrapper

__all__ = [
    'EnvSpec',
    'GarageEnv',
    'GridWorldEnv',
    'HalfCheetahDirEnv',
    'HalfCheetahVelEnv',
    'ML1WithPinnedGoal',
    'IgnoreDoneWrapper',
    'normalize',
    'normalize_reward',
    'PointEnv',
    'RL2Env',
    'Step',
    'TaskOnehotWrapper',
    'TaskIdWrapper',
    'TaskIdWrapper2',
]
