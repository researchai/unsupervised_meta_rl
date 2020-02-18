import gym


class TaskIdWrapper(gym.Wrapper):

    def __init__(self, env, task_id, task_name):

        super().__init__(env)
        self.task_id = task_id
        self.task_name = task_name

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['task_id'] = self.task_id
        info['task_name'] = self.task_name
        return obs, reward, done, info
