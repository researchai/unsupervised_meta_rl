import torch
from torch import nn
from garage.torch.modules import GaussianMLPModule


class GaussianMLPPolicy(nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 adaptive_std=False,
                 std_share_network=False,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_output_nonlinearity=None,
                 std_parameterization='exp',
                 layer_normalization=False):
        super(GaussianMLPPolicy, self).__init__()

        self._model = GaussianMLPModule(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            adaptive_std=adaptive_std,
            std_share_network=std_share_network,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_hidden_sizes=std_hidden_sizes,
            std_hidden_nonlinearity=std_hidden_nonlinearity,
            std_output_nonlinearity=std_output_nonlinearity,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization
        )

    def forward(self, inputs):
        return self._model(inputs)

    def get_action(self, env):
        action_var, mean, log_std_var, std, dist = self.forward(env)

        return action_var[0], dict(mean=mean[0], log_std=log_std_var[0])

    def get_actions(self, env):
        action_var, mean, log_std_var, std, dist = self.forward(env)

        return action_var, dict(mean=mean, log_std=log_std_var)
