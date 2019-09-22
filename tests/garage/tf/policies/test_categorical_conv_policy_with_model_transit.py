import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalConvPolicy
from garage.tf.policies import CategoricalConvPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv
from tests.fixtures.models import SimpleCNNModel
from tests.fixtures.models import SimpleMLPModel


class TestCategoricalConvPolicyWithModel(TfGraphTestCase):
    def setup_method(self):
        super().setup_method()
        env = TfEnv(DummyDiscretePixelEnv())
        self.default_filter_dims = (32, )
        self.default_filter_sizes = (3, )
        self.default_strides = (1, )
        self.default_padding = ('VALID',)
        self.default_hidden_sizes = (4, )
        self.default_initializer = tf.constant_initializer(0.5)

        self.policy1 = CategoricalConvPolicy(
            env_spec=env.spec,
            conv_filters=self.default_filter_dims,
            conv_filter_sizes=self.default_filter_sizes,
            conv_strides=self.default_strides,
            conv_pads=self.default_padding,
            hidden_sizes=self.default_hidden_sizes,
            hidden_w_init=self.default_initializer,
            output_w_init=self.default_initializer)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.policy2 = CategoricalConvPolicyWithModel(
            env_spec=env.spec,
            conv_filters=self.default_filter_dims,
            conv_filter_sizes=self.default_filter_sizes,
            conv_strides=self.default_strides,
            conv_pad=self.default_padding[0],
            hidden_sizes=self.default_hidden_sizes,
            hidden_w_init=self.default_initializer,
            output_w_init=self.default_initializer)

        self.obs = env.reset()

        self.obs_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, ) + env.observation_space.shape)

        self.dist1_sym = self.policy1.dist_info_sym(
            obs_var=self.obs_ph,
            name='p1_sym')

        self.dist2_sym = self.policy2.dist_info_sym(
            obs_var=self.obs_ph,
            name='p2_sym')

    def test_dist_info_sym_output(self):
        dist1 = self.sess.run(
            self.dist1_sym, feed_dict={self.obs_ph: [self.obs]})
        dist2 = self.sess.run(
            self.dist1_sym, feed_dict={self.obs_ph: [self.obs]})

        assert np.array_equal(dist1['prob'], dist2['prob'])
        assert np.array_equal(dist1['prob'], self.policy1.dist_info([self.obs])['prob'])
        assert np.array_equal(dist2['prob'], self.policy2.dist_info([self.obs])['prob'])

    @mock.patch('numpy.random.choice')
    def test_get_action(self, mock_rand):
        mock_rand.return_value = 0

        action1, agent_info1 = self.policy1.get_action(self.obs)
        action2, agent_info2 = self.policy2.get_action(self.obs)

        assert action1 == action2
        assert np.array_equal(agent_info1['prob'], agent_info2['prob'])

        actions1, agent_infos1 = self.policy1.get_actions([self.obs])
        actions2, agent_infos2 = self.policy2.get_actions([self.obs])

        assert np.array_equal(actions1, actions2)
        assert np.array_equal(agent_infos1['prob'], agent_infos2['prob'])
