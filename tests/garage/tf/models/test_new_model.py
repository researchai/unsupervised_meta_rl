import pickle

import numpy as np
import tensorflow as tf

from garage.tf.models.gaussian_mlp_model_ryan import GaussianMLPModel
from tests.fixtures import TfGraphTestCase
from garage.tf.core.mlp import mlp


class TestNewModel(TfGraphTestCase):
    def test_model_creation(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = GaussianMLPModel(output_dim=2)
        model.build(input_var)
        data = np.random.random((3, 5))
        self.sess.run(model.output, feed_dict={model.input: data})

    def test_same_parent_model_with_no_name(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = GaussianMLPModel(output_dim=2)
        # model2 = GaussianMLPModel(output_dim=2)
        model.build(input_var)
        # model2.build(input_var)

        tf.summary.FileWriter('log', self.sess.graph)
        for x in tf.global_variables():
            print(x)

    def test_same_model_with_no_name(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = GaussianMLPModel(output_dim=2)
        model.build(input_var)
        with self.assertRaises(ValueError):
            model.build(input_var)

    def test_model_with_different_name(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = GaussianMLPModel(output_dim=2)
        model.build(input_var)
        model.build(input_var, name="model_2")
        data = np.random.random((3, 5))
        results, results2 = self.sess.run([model.outputs[:2], model.model_2.outputs[:2]], feed_dict={input_var: data})
        assert np.array_equal(results, results2)

    def test_model_with_different_name_in_different_order(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = GaussianMLPModel(output_dim=2)
        model.build(input_var, name="model_2")
        model.build(input_var)
        data = np.random.random((3, 5))
        results, results2 = self.sess.run([model.outputs[:2], model.model_2.outputs[:2]], feed_dict={input_var: data})
        assert np.array_equal(results, results2)

    def test_model_pickle(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = GaussianMLPModel(output_dim=2)
        model.build(input_var)
        data = np.random.random((3, 5))
        results = self.sess.run(model.outputs[:2], feed_dict={input_var: data})

        model_pickled = pickle.loads(pickle.dumps(model))
        model_pickled.build(input_var)
        results2 = self.sess.run(model_pickled.outputs[:2], feed_dict={input_var: data})

        assert np.array_equal(results, results2)
        assert isinstance(model.dist, tf.contrib.distributions.MultivariateNormalDiag)

    def test_model_child_cannot_be_built(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = GaussianMLPModel(output_dim=2)
        model.build(input_var, name="child")
        with self.assertRaises(ValueError):
            model.child.build(input_var, name="grandchild")

    def test_model_child_not_pickable(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = GaussianMLPModel(output_dim=2)
        model.build(input_var, name="child")
        with self.assertRaises(ValueError):
            pickle.loads(pickle.dumps(model.child))


