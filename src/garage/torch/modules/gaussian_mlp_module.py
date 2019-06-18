import torch
from torch import nn
from torch.distributions import MultivariateNormal

from garage.torch.modules import MLPModule


class GaussianMLPModule(nn.Module):
    """
    GaussianMLPModel.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
        adaptive_std (bool): Is std a neural network. If False, it will be a
            parameter.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues.
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues.
        std_hidden_nonlinearity: Nonlinearity for each hidden layer in
            the std network.
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a tf.Tensor. Set
            it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parametrization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
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
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False):
        super(GaussianMLPModule, self).__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_std = learn_std
        self._adaptive_std = adaptive_std
        self._std_share_network = std_share_network
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        # Tranform std arguments to parameterized space
        self._init_std_param, self._min_std_param, self._max_std_param = None, None, None

        init_std = torch.Tensor([init_std])
        if min_std:
            min_std = torch.Tensor([min_std])
        if max_std:
            max_std = torch.Tensor([max_std])

        if self._std_parameterization == 'exp':
            self._init_std_param = init_std.log()
            self._min_std_param = min_std.log() if min_std else None
            self._max_std_param = max_std.log() if max_std else None

        elif self._std_parameterization == 'softplus':
            self._init_std_param = init_std.exp().add(-1).log()
            self._min_std_param = min_std.exp().add(-1).log() if min_std else None
            self._max_std_param = max_std.exp().add(-1).log() if max_std else None

        else:
            raise NotImplementedError

        if std_share_network:
            self.shared_mean_std_network = self._get_shared_mean_std_network()
        else:
            self.mean_module = self._get_mean_network()

            if adaptive_std:
                self.adaptive_std_module = self._get_adaptive_std_network()

    def _get_shared_mean_std_network(self):
        return MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim * 2,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._init_shared_b,
            layer_normalization=self._layer_normalization
        )

    def _init_shared_b(self, b):
        nn.init.zeros_(b[..., :self._action_dim])
        nn.init.constant_(b[..., self._action_dim:], self._init_std_param.item())

    def _get_mean_network(self):
        return MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization)

    def _get_adaptive_std_network(self):
        return MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._std_hidden_sizes,
            hidden_nonlinearity=self._std_hidden_nonlinearity,
            hidden_w_init=self._std_hidden_w_init,
            hidden_b_init=self._std_hidden_b_init,
            output_nonlinearity=self._std_output_nonlinearity,
            output_w_init=self._std_output_w_init,
            output_b_init=self._init_adaptive_b,
            layer_normalization=self._layer_normalization)

    def _init_adaptive_b(self, b):
        return nn.init.constant_(b, self._init_std_param.item())

    def forward(self, inputs):
        if self._std_share_network:
            output = self.shared_mean_std_network(inputs)
            mean = output[..., :self._action_dim]
            std = output[..., self._action_dim:]
        else:
            mean = self.mean_module(inputs)

            if self._adaptive_std:
                std = self.adaptive_std_module(inputs)
            else:
                broadcast_shape = list(inputs.shape[:-1]) + [self._action_dim]
                x = torch.zeros(*broadcast_shape)
                std = x + self._init_std_param

        if self._min_std_param or self._max_std_param:
            std = std.clamp(min=to_scalar(self._min_std_param), max=to_scalar(self._max_std_param))

        if self._std_parameterization == 'exp':
            log_std_var = std
        else:
            log_std_var = std.exp().add(1.).log().log()

        rnd = torch.rand(list(mean.shape)[1:])
        action_var = rnd * log_std_var.exp() + mean

        dist = MultivariateNormal(mean, torch.eye(self._action_dim) * (log_std_var.exp()**2))

        return action_var, mean, log_std_var, std, dist


def to_scalar(tensor):
    return None if tensor is None else tensor.item()
