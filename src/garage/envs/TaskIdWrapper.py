import gym


class TaskIdWrapper(gym.Wrapper):
    @property
    def _hidden_env(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    @property
    def task_names(self):
        return self._hidden_env._task_names

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['task_id'] = self.task_id
        info['task_name'] = self.task_name
        return obs, reward, done, info

    def set_task(self, task):
        self.env.set_task(task)
        self.task_id = self._hidden_env._active_task
        self.task_name = self._hidden_env._task_names[self.task_id]
