import numpy as np
import torch

import garage.torch.utils as tu
from garage.torch.modules import MLPModule
from garage.torch.policies import Policy


class CategoricalMLPPolicy(Policy, MLPModule):
    def __init__(self, env_spec, name="CategoricalMLPPolicy", **kwargs):
        self._obs_dim = env_spec.input_space.flat_dim
        self._action_dim = env_spec.output_space.flat_dim

        Policy.__init__(self, env_spec, name)
        MLPModule.__init__(input_dim=self._obs_dim,
                           output_dim=self._action_dim,
                           **kwargs)

    def forward(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states).float().to(
                tu.global_device())
        x = super().forward(states)
        return torch.softmax(x, dim=-1)

    def get_actions(self, states):
        with torch.no_grad():
            # if not isinstance(states, torch.Tensor):
            #    states = torch.from_numpy(states).float().to(
            #        tu.global_device())
            x = self.forward(torch.Tensor(states))
            return np.array([np.random.choice(self._action_dim, p=x.numpy()[idx])
                             for idx in range(x.numpy().shape[0])])

    def get_action(self, state):
        with torch.no_grad():
            # if not isinstance(state, torch.Tensor):
            #    states = torch.from_numpy(state).float().to(
            #        tu.global_device())
            x = self.forward(torch.Tensor(state).unsqueeze(0))
            return np.random.choice(x.squeeze(0).numpy(), p=x.squeeze(0).numpy())
