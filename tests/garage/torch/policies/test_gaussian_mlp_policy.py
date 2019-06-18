import pickle
from garage.torch.policies import GaussianMLPPolicy
import unittest
from unittest import mock
import torch
from nose2.tools.params import params


class TestGaussianMLPPolicy(unittest.TestCase):

    @mock.patch('garage.torch.policies.gaussian_mlp_policy.GaussianMLPModule')
    def test_policy_get_actions(self, mock_model):
        action, mean, log_std = (torch.Tensor(x) for x in ([5.0, 3., 2.], [0.5, 0.2, 0.4], [0.25, 0.11, 0.44]))
        mock_model.return_value = lambda x: (action, mean, log_std, None, None)
        input_dim, output_dim, hidden_sizes = (5, 3, (2, 2))

        policy = GaussianMLPPolicy(input_dim, output_dim, hidden_sizes=hidden_sizes)

        input = torch.ones(input_dim)
        sample, dist_info = policy.get_actions(input)

        assert sample.equal(action)
        assert dist_info['mean'].equal(mean)
        assert dist_info['log_std'].equal(log_std)

    @mock.patch('garage.torch.policies.gaussian_mlp_policy.GaussianMLPModule')
    def test_policy_get_action(self, mock_model):
        action, mean, log_std = (torch.Tensor(x) for x in ([5.0, 3., 2.], [0.5, 0.2, 0.4], [0.25, 0.11, 0.44]))
        mock_model.return_value = lambda x: (action, mean, log_std, None, None)
        input_dim, output_dim, hidden_sizes = (5, 3, (2, 2))

        policy = GaussianMLPPolicy(input_dim, output_dim, hidden_sizes=hidden_sizes)

        input = torch.ones(input_dim)
        sample, dist_info = policy.get_action(input)

        assert sample.equal(action[0])
        assert dist_info['mean'].equal(mean[0])
        assert dist_info['log_std'].equal(log_std[0])

    @mock.patch('torch.rand')
    @params((5, 1, (1,)), (5, 1, (2,)), (5, 2, (3,)), (5, 2, (1, 1)), (5, 3, (2, 2)))
    def test_policy_is_picklable(self, input_dim, output_dim, hidden_sizes, mock_normal):
        mock_normal.return_value = 0.5

        policy = GaussianMLPPolicy(input_dim, output_dim, hidden_sizes=hidden_sizes)

        input = torch.ones(input_dim)
        sample, dist_info = policy.get_actions(input)

        h = pickle.dumps(policy)
        policy_pickled = pickle.loads(h)

        sample_p, dist_info_p = policy_pickled.get_actions(input)

        assert sample.equal(sample_p)
        assert dist_info['mean'].equal(dist_info_p['mean'])
        assert dist_info['log_std'].equal(dist_info_p['log_std'])
