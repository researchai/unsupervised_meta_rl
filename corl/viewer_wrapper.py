from types import MethodType

import cv2
import gym
import numpy as np

from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_open_6dof import SawyerBoxOpen6DOFEnv


FOUR_GOOD_ANGLES = [
    (np.array([ 0.07299577,  0.32976531, -0.16484401]), 2.4713219507278814, -32.48425196850392, -50.13779527559039),
    (np.array([-0.14497756,  0.2378877 , -0.07864494]), 2.4180270187662978, -18.07086614173229, -133.58267716535477),
    (np.array([-0.08665162,  0.31736463, -0.19983442]), 2.517476543670253, -36.14173228346461, 147.7559055118108),
    (np.array([-0.08495986,  0.40854054, -0.14935483]), 2.4842945432918873, -34.72440944881881, -178.2283464566922),
]

class ScreenShotWrapper(gym.Wrapper):

    def __init__(self, env, setting_values, name='env'):
        super().__init__(env,)
        self.name = name
        setting_keys = ['lookat', 'distance', 'elevation', 'azimuth']

        viewer_settings = []
        
        for s in setting_values:
            setting = dict(
                    lookat=s[0],
                    distance=s[1],
                    elevation=s[2],
                    azimuth=s[3],
                )
            print(setting)
            viewer_settings.append(setting)

        self.viewer_settings = viewer_settings

    def set_inner_env_viewer(self, setting):
        def _viewer_setup(env):
            env.viewer.cam.trackbodyid = 0
            env.viewer.cam.lookat[0] = setting['lookat'][0]
            env.viewer.cam.lookat[1] = setting['lookat'][1]
            env.viewer.cam.lookat[2] = setting['lookat'][2]
            env.viewer.cam.distance = setting['distance']
            env.viewer.cam.elevation = setting['elevation']
            env.viewer.cam.azimuth = setting['azimuth']
            env.viewer.cam.trackbodyid = -1
        self.env.viewer_setup = MethodType(_viewer_setup, self.env)


    def render(self):
        self.env.render()
        idx = 0
        for i, setting in enumerate(self.viewer_settings):
            self.set_inner_env_viewer(setting)
            image = np.array(self.env.render(mode='rgb_array'))
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            img[:, :, 0] = image[:, :, 2]
            img[:, :, 1] = image[:, :, 1]
            img[:, :, 2] = image[:, :, 0]
            cv2.imwrite('./{}_{}.png'.format(self.name, i), img)


if __name__ == '__main__':
    env = SawyerBoxOpen6DOFEnv()
    wrapper = ScreenShotWrapper(env, FOUR_GOOD_ANGLES)
    wrapper.render()
    