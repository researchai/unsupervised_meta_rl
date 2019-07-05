import numpy as np

from multiworld.envs.mujoco.sawyer_xyz.env_lists import HARD_MODE_LIST
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_wall_6dof import SawyerReachPushPickPlaceWall6DOFEnv


TRAIN_DICT = {
    i: env
    for i, env in enumerate(HARD_MODE_LIST)
}


_reach_push_pick_place = 0
_reach_push_pick_place_wall = 0

def hard_mode_args_kwargs(env_cls):
    global _reach_push_pick_place, _reach_push_pick_place_wall
    kwargs = dict(random_init=True, obs_type='with_goal', if_render=False)
    if env_cls == SawyerReachPushPickPlace6DOFEnv:
        assert _reach_push_pick_place <= 2
        if _reach_push_pick_place == 0:
            kwargs['tasks'] = [{
                'goal': np.array([-0.1, 0.8, 0.2]),
                'obj_init_pos':np.array([0, 0.6, 0.02]),
                'obj_init_angle': 0.3,
                'type':'reach',}]
        elif _reach_push_pick_place == 1:
            kwargs['tasks'] = [{
                'goal': np.array([0.1, 0.8, 0.02]),
                'obj_init_pos':np.array([0, 0.6, 0.02]),
                'obj_init_angle': 0.3,
                'type':'push',}]
        else:
            kwargs['tasks'] = [{
                'goal': np.array([0.1, 0.8, 0.2]),
                'obj_init_pos':np.array([0, 0.6, 0.02]),
                'obj_init_angle': 0.3,
                'type': 'pick_place',}]
        _reach_push_pick_place += 1
    elif env_cls == SawyerReachPushPickPlaceWall6DOFEnv:
        assert _reach_push_pick_place_wall <= 2
        if _reach_push_pick_place_wall == 0:
            kwargs['tasks'] = [{
                'goal': np.array([0.05, 0.8, 0.2]),
                'obj_init_pos':np.array([0, 0.6, 0.015]),
                'obj_init_angle': 0.3,
                'type':'pick_place'},]
        elif _reach_push_pick_place_wall == 1:
            kwargs['tasks'] = [{
                'goal': np.array([-0.05, 0.8, 0.2]),
                'obj_init_pos':np.array([0, 0.6, 0.015]),
                'obj_init_angle': 0.3,
                'type': 'reach'},]
        else:
            kwargs['tasks'] = [{
                'goal': np.array([0.05, 0.8, 0.015]),
                'obj_init_pos':np.array([0, 0.6, 0.015]),
                'obj_init_angle': 0.3,
                'type': 'push',}]
        _reach_push_pick_place_wall += 1
    return dict(args=[], kwargs=kwargs)


TRAIN_ARGS_KWARGS = {
    i: hard_mode_args_kwargs(env)
    for i, env in enumerate(HARD_MODE_LIST)
}
