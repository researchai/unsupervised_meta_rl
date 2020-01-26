"""Test for the MutliEnvWrapper."""
from metaworld.benchmarks import MT10

from metaworld.envs.mujoco.env_dict import EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT
from garage.envs import GarageEnv
from garage.envs.multi_task_metaworld_wrapper import MTMetaWorldWrapper

class TestMetaWorldWrapper:
    """Tests for garage.envs.multi_task_metaworld_wrapper."""

    def test_env_creation(self):
        """Test env creation."""
        MT10_envs_by_id = {}

        for (task, env) in EASY_MODE_CLS_DICT.items():
            MT10_envs_by_id[task] = GarageEnv(env(*EASY_MODE_ARGS_KWARGS[task]['args'],
                                        **EASY_MODE_ARGS_KWARGS[task]['kwargs']))

        envs = MTMetaWorldWrapper(MT10_envs_by_id)

    def test_env_task_name_to_one_hot(self):
        MT10_envs_by_id = {}

        for (task, env) in EASY_MODE_CLS_DICT.items():
            MT10_envs_by_id[task] = GarageEnv(env(*EASY_MODE_ARGS_KWARGS[task]['args'],
                                        **EASY_MODE_ARGS_KWARGS[task]['kwargs']))

        envs = MTMetaWorldWrapper(MT10_envs_by_id)
        print(envs.task_name_to_one_hot)