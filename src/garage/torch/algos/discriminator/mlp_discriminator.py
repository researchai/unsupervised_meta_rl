import torch
import numpy as np
from gym.envs import kwargs

from garage.torch.modules import MLPModule


class MLPDiscriminator(MLPModule):
    def __init__(self, env_spec, skills_num):
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._skills_num = skills_num

        super().__init__(input_dim=self._obs_dim,
                         output_dim=skills_num,
                         **kwargs)

    def forward(self, states):
        x = super().forward(states)
        return torch.softmax(x, dim=-1)

    def infer_skills(self, states):
        with torch.no_grad():
            x = self.forward(torch.Tensor(states))
            return np.array([np.random.choice(self._skills_num, p=x.numpy()[idx])
                             for idx in range(x.numpy().shape[0])])

    def infer_skill(self, state):
        with torch.no_grad():
            x = self.forward(torch.Tensor(state).unsqueeze(0))
            return np.random.choice(x.squeeze(0).numpy(), p=x.squeeze(0).numpy())
