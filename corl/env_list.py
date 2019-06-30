import numpy as np

from multiworld.envs.mujoco.sawyer_xyz.env_lists import EASY_MODE_LIST
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_6dof import SawyerDoor6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_stack_6dof import SawyerStack6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsert6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg_6dof import SawyerNutAssembly6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweep6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_window_open_6dof import SawyerWindowOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_hammer_6dof import SawyerHammer6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_window_close_6dof import SawyerWindowClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn_6dof import SawyerDialTurn6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPull6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open_6dof import SawyerDrawerOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_6dof import SawyerButtonPressTopdown6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close_6dof import SawyerDrawerClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_open_6dof import SawyerBoxOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_close_6dof import SawyerBoxClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side_6dof import SawyerPegInsertionSide6DOFEnv


EASY_MODE_DICT = {
    'reach': SawyerReachPushPickPlace6DOFEnv,
    'push': SawyerReachPushPickPlace6DOFEnv,
    'pickplace': SawyerReachPushPickPlace6DOFEnv,
    'door_open': SawyerDoor6DOFEnv,
    'drawer_open': SawyerDrawerOpen6DOFEnv,
    'drawer_close': SawyerDrawerClose6DOFEnv,
    'button_press_top_down': SawyerButtonPressTopdown6DOFEnv,
    'peg_insertion_side': SawyerPegInsertionSide6DOFEnv,
    'window_open': SawyerWindowOpen6DOFEnv,
    'window_close': SawyerWindowClose6DOFEnv,
}


EASY_MODE_ARGS_KWARGS = {
    'reach': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([-0.1, 0.8, 0.2]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'reach'}],
            'random_init': False,
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
        }
    },
    'push': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0.1, 0.8, 0.02]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'push'}],
            'random_init': False,
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
        }
    },
    'pickplace': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0.1, 0.8, 0.2]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'pick_place'}],
            'random_init': False,
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
        }
    },
    'door_open': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([-0.2, 0.7, 0.15]),  'obj_init_pos':np.array([0.1, 0.95, 0.1]), 'obj_init_angle': 0.3}],
            'random_init': False,
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
        }
    },
    'drawer_open': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0., 0.55, 0.04]),  'obj_init_pos':np.array([0., 0.9, 0.04]), 'obj_init_angle': 0.3}],
            'random_init': False,
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
        }
    },
    'drawer_close': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0., 0.7, 0.04]),  'obj_init_pos':np.array([0., 0.9, 0.04]), 'obj_init_angle': 0.3}],
            'random_init': False,
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
        }
    },
    'button_press_top_down': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0, 0.88, 0.1]), 'obj_init_pos':np.array([0, 0.8, 0.05])}],
            'random_init': False,
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
        }
    },
    'peg_insertion_side': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([-0.3, 0.6, 0.05]), 'obj_init_pos':np.array([0, 0.6, 0.02])}],
            'random_init': False,
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
        }
    },
    'window_open': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0.08, 0.785, 0.15]),  'obj_init_pos':np.array([-0.1, 0.785, 0.15]), 'obj_init_angle': 0.3}],
            'random_init': False,
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
        }
    },
    'window_close': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([-0.08, 0.785, 0.15]),  'obj_init_pos':np.array([0.1, 0.785, 0.15]), 'obj_init_angle': 0.3}],
            'random_init': False,
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
        }
    },
}
