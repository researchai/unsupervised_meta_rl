import gym
import numpy as np

class TaskIdWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

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

    def sample_tasks(self, meta_batch_size):
        if self._hidden_env._sampled_all:
            assert meta_batch_size >= len(self._hidden_env._task_envs)
            tasks = [i for i in range(meta_batch_size)]
        else:
            tasks = np.random.choice(
                self._hidden_env.num_tasks, size=meta_batch_size, replace=False).tolist()
        if self._hidden_env._sample_goals:
            goals = [
                self._hidden_env._task_envs[t % len(self._hidden_env._task_envs)].sample_goals_(1)[0]
                for t in tasks
            ]
            tasks_with_goal = [
                dict(task=t, goal=g)
                for t, g in zip(tasks, goals)
            ]
            return tasks_with_goal
        else:
            return tasks

