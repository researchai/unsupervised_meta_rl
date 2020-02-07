import gym


class TaskIdWrapper(gym.Wrapper):
    @property
    def _hidden_env(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    @property
    def task_id(self):
        return self._hidden_env._active_task

    @property
    def task_name(self):
        return self._hidden_env._task_names[self.task_id]

    @property
    def task_names(self):
        return self._hidden_env._task_names

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if hasattr(self, 'task_id'):
            info['task_id'] = self.task_id
        if hasattr(self, 'task_name'):
            info['task_name'] = self.task_name
        return obs, reward, done, info

    def set_task(self, task):
        self.env.set_task(task)
