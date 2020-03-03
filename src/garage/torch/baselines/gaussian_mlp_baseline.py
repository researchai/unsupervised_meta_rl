"""A value function (baseline) based on a GaussianMLP model."""
import numpy as np
import torch
from dowel import tabular
from torch import nn

from garage.np.baselines import Baseline
from garage.torch.algos import make_optimizer
from garage.torch.modules import GaussianMLPModule
from torch.distributions import Independent, Normal


class GaussianMLPBaseline(Baseline):
    """Gaussian MLP Baseline with Model.

    It fits the input data to a gaussian distribution estimated by
    a MLP.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normalize_observation (bool): Bool for normalizing observation or not.
        normalize_return (bool): Bool for normalizing return or not.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in torch.optim.
        optimizer_args (dict): Optimizer arguments.
    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 layer_normalization=False,
                 normalize_observation=True,
                 normalize_return=True,
                 optimizer=torch.optim.Adam,
                 optimizer_args=None,
                 minibatch_size=128,
                 max_optimization_epochs=10,
                 name='GaussianMLPBaseline'):
        super().__init__(env_spec)

        input_dim = env_spec.observation_space.flat_dim
        output_dim = 1

        self._module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=None,
            max_std=None,
            std_parameterization='exp',
            layer_normalization=layer_normalization)

        self._normalize_observation = normalize_observation
        self._normalize_return = normalize_return
        self._name = name
        self._obs_dist = dict(mean=np.zeros(input_dim), std=np.ones(input_dim))
        self._return_dist = dict(mean=np.zeros(output_dim), std=np.ones(output_dim))
        self._minibatch_size = minibatch_size
        self._max_optimization_epochs = max_optimization_epochs

        if optimizer_args is None:
            optimizer_args = dict(lr=3e-4)

        self._optimizer = make_optimizer(optimizer, self._module,
                                         **optimizer_args)

    def fit(self, paths):
        """Fit regressor based on paths.

        Args:
            paths (list[dict]): Sample paths.

        """
        observations = np.concatenate([p['observations'] for p in paths])
        returns = np.concatenate([p['returns'] for p in paths]).reshape(-1, 1)

        if self._normalize_observation:
            mean = np.mean(observations, axis=0, keepdims=True)
            std = np.std(observations, axis=0, keepdims=True) + 1e-8
            self._obs_dist = dict(mean=mean, std=std)

        if self._normalize_return:
            mean = np.mean(returns, axis=0, keepdims=True)
            std = np.std(returns, axis=0, keepdims=True) + 1e-8
            self._return_dist = dict(mean=mean, std=std)

        normalized_obs = torch.Tensor((observations - self._obs_dist['mean']) /
                                      self._obs_dist['std'])
        normalized_return = torch.Tensor((returns - self._return_dist['mean']) /
                                         self._return_dist['std'])

        with torch.no_grad():
            dist = self._module(normalized_obs)
            loss_before = -dist.log_prob(
                torch.Tensor(normalized_return)).mean()

        step_size = (self._minibatch_size
                     if self._minibatch_size else len(returns))

        for epoch in range(self._max_optimization_epochs):
            rand_ids = np.random.permutation(len(returns))
            for start_idx in range(0, len(returns), step_size):
                ids = rand_ids[start_idx:start_idx + step_size]
                self._fit(normalized_obs[ids], normalized_return[ids])

        with torch.no_grad():
            dist = self._module(normalized_obs)
            loss_after = -dist.log_prob(torch.Tensor(normalized_return)).mean()

        with tabular.prefix(self._name):
            tabular.record('/LossBefore', loss_before.item())
            tabular.record('/LossAfter', loss_after.item())
            tabular.record('/dLoss', loss_before.item() - loss_after.item())

    def _fit(self, normalized_obs, normalized_return):
        dist = self._module(normalized_obs)
        ll = dist.log_prob(normalized_return)
        loss = -ll.mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def predict(self, path):
        """Predict value based on paths.

        Args:
            path (list[dict]): Sample paths.

        Returns:
            numpy.ndarray: Predicted value.

        """
        normalized_obs = torch.Tensor((path['observations'] - self._obs_dist['mean']) /
                                      self._obs_dist['std'])
        with torch.no_grad():
            dist = self._module(normalized_obs)
        normalized_mean = dist.mean.flatten().numpy()
        mean = normalized_mean * self._return_dist['std'] + self._return_dist['mean']
        return mean.flatten()

    def get_param_values(self):
        """Get the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Returns:
            dict: The parameters (in the form of the state dictionary).

        """
        return self._module.state_dict()

    def set_param_values(self, flattened_params):
        """Set the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Args:
            flattened_params (dict): State dictionary.

        """
        self._module.load_state_dict(flattened_params)
