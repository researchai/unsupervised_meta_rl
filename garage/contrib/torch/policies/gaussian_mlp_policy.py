import torch
import torch.nn as nn

from garage.contrib.torch.core.mlp import mlp

class GaussianMLPPolicy():
    def __init__(self,
                 env_spec,
                 hidden_sizes=(32,32),
                 hidden_nonlinearity=torch.tanh,
                 output_nonlinearity=torch.tanh,
                 learn_std=False):
        if learn_std:
            raise NotImplementedError

        self.mu = mlp()

