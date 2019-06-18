import pickle
from garage.torch.modules import GaussianMLPModule
import unittest
from unittest import mock
import torch
from torch import nn
from nose2.tools.params import params


class TestGaussianMLPmodule(unittest.TestCase):

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_std_share_network_output_values(self, input_dim, output_dim, hidden_sizes, mock_normal):
        mock_normal.return_value = 0.5

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            std_share_network=True,
            hidden_nonlinearity=None,
            std_parameterization='exp',
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_
        )

        action, mean, log_std, std, dist = module(torch.ones(5))

        assert mean.equal(5 * (torch.Tensor(hidden_sizes).prod() + torch.zeros(output_dim)))
        assert std.equal(5 * (torch.Tensor(hidden_sizes).prod() + torch.zeros(output_dim)))
        assert log_std.equal(5 * (torch.Tensor(hidden_sizes).prod() + torch.zeros(output_dim)))
        assert action.equal((0.5 * log_std.exp() + mean) + torch.zeros(output_dim))

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_std_network_output_values(self, input_dim, output_dim, hidden_sizes, mock_rand):
        mock_rand.return_value = 0.5

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            init_std=2,
            std_share_network=False,
            adaptive_std=False,
            hidden_nonlinearity=None,
            std_parameterization='exp',
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_
        )

        action, mean, log_std, std, dist = module(torch.ones(5))

        assert mean.equal(5 * (torch.Tensor(hidden_sizes).prod() + torch.zeros(output_dim)))
        assert std.equal(torch.Tensor([2]).log() + torch.zeros(output_dim))
        assert log_std.equal(torch.Tensor([2]).log() + torch.zeros(output_dim))
        assert action.equal((0.5 * log_std.exp() + mean) + torch.zeros(output_dim))
        assert dist.mean.equal(mean)
        assert dist.variance.equal(log_std.exp()**2)

    @mock.patch('torch.rand')
    @params((5, 1, (1,), (1,)), (5, 1, (2,), (2,)), (5, 2, (3,), (3,)), (5, 2, (1, 1), (1, 1)), (5, 3, (2, 2), (2, 2)))
    def test_std_adaptive_network_output_values(self, input_dim, output_dim, hidden_sizes, std_hidden_sizes, mock_rand):
        mock_rand.return_value = 0.5

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            std_hidden_sizes=std_hidden_sizes,
            std_share_network=False,
            adaptive_std=True,
            hidden_nonlinearity=None,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_,
            std_hidden_nonlinearity=None,
            std_hidden_w_init=nn.init.ones_,
            std_output_w_init=nn.init.ones_
        )

        action, mean, log_std, std, dist = module(torch.ones(5))

        assert mean.equal(5 * (torch.Tensor(hidden_sizes).prod() + torch.zeros(output_dim)))
        assert std.equal(5 * (torch.Tensor(hidden_sizes).prod() + torch.zeros(output_dim)))
        assert log_std.equal(5 * (torch.Tensor(hidden_sizes).prod() + torch.zeros(output_dim)))
        assert action.equal((0.5 * log_std.exp() + mean) + torch.zeros(output_dim))
        assert dist.mean.equal(mean)
        assert dist.variance.equal(log_std.exp()**2)

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_softplus_std_network_output_values(self, input_dim, output_dim, hidden_sizes, mock_rand):
        mock_rand.return_value = 0.5

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            init_std=2,
            std_share_network=False,
            adaptive_std=False,
            hidden_nonlinearity=None,
            std_parameterization='softplus',
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_
        )

        action, mean, log_std, std, dist = module(torch.ones(5))

        assert mean.equal(5 * (torch.Tensor(hidden_sizes).prod() + torch.zeros(output_dim)))
        assert std.equal(torch.Tensor([2]).exp().add(-1).log() + torch.zeros(output_dim))
        assert log_std.equal(std.exp().add(1.).log().log() + torch.zeros(output_dim))
        assert action.equal((0.5 * log_std.exp() + mean) + torch.zeros(output_dim))
        assert dist.mean.equal(mean)
        assert dist.variance.equal(log_std.exp()**2)

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_exp_min_std(self, input_dim, output_dim, hidden_sizes, mock_rand):
        mock_rand.return_value = 0.5

        min_value = 10

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            init_std=1,
            min_std=min_value,
            std_share_network=False,
            adaptive_std=False,
            hidden_nonlinearity=None,
            std_parameterization='exp',
            hidden_w_init=nn.init.zeros_,
            output_w_init=nn.init.zeros_
        )

        action, mean, log_std, std, dist = module(torch.ones(5))

        assert std.equal(torch.Tensor([min_value]).log() + torch.zeros(output_dim))
        assert log_std.equal(torch.Tensor([min_value]).log() + torch.zeros(output_dim))

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_exp_max_std(self, input_dim, output_dim, hidden_sizes, mock_rand):
        mock_rand.return_value = 0.5

        max_value = 1

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            init_std=10,
            max_std=max_value,
            std_share_network=False,
            adaptive_std=False,
            hidden_nonlinearity=None,
            std_parameterization='exp',
            hidden_w_init=nn.init.zeros_,
            output_w_init=nn.init.zeros_
        )

        action, mean, log_std, std, dist = module(torch.ones(5))

        assert std.equal(torch.Tensor([max_value]).log() + torch.zeros(output_dim))
        assert log_std.equal(torch.Tensor([max_value]).log() + torch.zeros(output_dim))

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_softplus_min_std(self, input_dim, output_dim, hidden_sizes, mock_rand):
        mock_rand.return_value = 0.5

        min_value = 10

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            init_std=1,
            min_std=min_value,
            std_share_network=False,
            adaptive_std=False,
            hidden_nonlinearity=None,
            std_parameterization='softplus',
            hidden_w_init=nn.init.zeros_,
            output_w_init=nn.init.zeros_
        )

        action, mean, log_std, std, dist = module(torch.ones(5))

        assert std.equal(torch.Tensor([min_value]).exp().add(-1).log() + torch.zeros(output_dim))
        assert log_std.equal(std.exp().add(1.).log().log())

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_softplus_max_std(self, input_dim, output_dim, hidden_sizes, mock_rand):
        mock_rand.return_value = 0.5

        max_value = 1

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            init_std=10,
            max_std=max_value,
            std_share_network=False,
            adaptive_std=False,
            hidden_nonlinearity=None,
            std_parameterization='softplus',
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_
        )

        action, mean, log_std, std, dist = module(torch.ones(5))

        assert std.equal(torch.Tensor([max_value]).exp().add(-1).log() + torch.zeros(output_dim))
        assert log_std.equal(std.exp().add(1.).log().log())

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_std_network_output_values_pickable(self, input_dim, output_dim, hidden_sizes, mock_rand):
        mock_rand.return_value = 0.5

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            init_std=2,
            std_share_network=False,
            adaptive_std=False,
            hidden_nonlinearity=None,
            std_parameterization='exp',
        )

        input = torch.ones(5)
        action, mean, log_std, std, dist = module(input)

        h = pickle.dumps(module)
        module_pickled = pickle.loads(h)

        action_p, mean_p, log_std_p, std_p, dist_p = module_pickled(input)

        assert mean.equal(mean_p)
        assert std.equal(std_p)
        assert log_std.equal(log_std_p)
        assert action.equal(action_p)
        assert dist.mean.equal(dist_p.mean)
        assert dist.variance.equal(dist_p.variance)

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_std_adaptive_network_output_values_pickable(self, input_dim, output_dim, hidden_sizes, mock_rand):
        mock_rand.return_value = 0.5

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            init_std=2,
            std_share_network=False,
            adaptive_std=True,
            hidden_nonlinearity=None,
            std_parameterization='exp',
        )

        input = torch.ones(5)
        action, mean, log_std, std, dist = module(input)

        h = pickle.dumps(module)
        module_pickled = pickle.loads(h)

        action_p, mean_p, log_std_p, std_p, dist_p = module_pickled(input)

        assert mean.equal(mean_p)
        assert std.equal(std_p)
        assert log_std.equal(log_std_p)
        assert action.equal(action_p)
        assert dist.mean.equal(dist_p.mean)
        assert dist.variance.equal(dist_p.variance)

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_std_shared_network_output_values_pickable(self, input_dim, output_dim, hidden_sizes, mock_rand):
        mock_rand.return_value = 0.5

        module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            init_std=2,
            std_share_network=True,
            hidden_nonlinearity=None,
            std_parameterization='exp',
        )

        input = torch.ones(5)
        action, mean, log_std, std, dist = module(input)

        h = pickle.dumps(module)
        module_pickled = pickle.loads(h)

        action_p, mean_p, log_std_p, std_p, dist_p = module_pickled(input)

        assert mean.equal(mean_p)
        assert std.equal(std_p)
        assert log_std.equal(log_std_p)
        assert action.equal(action_p)
        assert dist.mean.equal(dist_p.mean)
        assert dist.variance.equal(dist_p.variance)

    def test_unknown_std_parameterization(self):
        with self.assertRaises(NotImplementedError):
            GaussianMLPModule(input_dim=1, output_dim=1, std_parameterization='unknown')
