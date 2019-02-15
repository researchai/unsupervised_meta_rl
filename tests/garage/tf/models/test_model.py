import pickle

import numpy as np
import tensorflow as tf

from garage.tf.core.mlp import mlp
from garage.tf.models.base import TfModel
from tests.fixtures import TfGraphTestCase


class SimpleModel(TfModel):
    def __init__(self, output_dim=2, hidden_sizes=(4, 4), name=None):
        super().__init__(name)
        self._output_dim = output_dim
        self._hidden_sizes = hidden_sizes

    def _build(self, obs_input):
        mlp1 = mlp(obs_input,
                   self._output_dim,
                   self._hidden_sizes,
                   "mlp1")

        return mlp1

    @property
    def init_spec(self):
        return (), {
            'name': self._name,
            'output_dim': self._output_dim,
            'hidden_sizes': self._hidden_sizes
        }

class ComplicatedModel(TfModel):
    def __init__(self, output_dim=2, name=None):
        super().__init__(name)
        self._output_dim = output_dim
        self._simple_model_1 = SimpleModel(output_dim=4)
        self._simple_model_2 = SimpleModel(output_dim=output_dim)

    def _build(self, obs_input):
        h1 = self._simple_model_1.build(obs_input)
        return self._simple_model_2.build(h1)

    @property
    def init_spec(self):
        return (), {
            'name': self._name,
            'output_dim': self._output_dim
        }


class TestModel(TfGraphTestCase):
    def test_model_creation(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        model.build(input_var)
        data = np.ones((3, 5))
        self.sess.run(model.output, feed_dict={model.input: data})

    def test_model_creation_with_custom_name(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        model.build(input_var, name='custom_model')
        data = np.ones((3, 5))
        result, result2 = self.sess.run(
            [model.outputs, model.custom_model.outputs],
            feed_dict={model.input: data})
        assert np.array_equal(result, result2)

    def test_same_parent_model_with_no_name(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        model2 = SimpleModel(output_dim=2)
        model.build(input_var)
        with self.assertRaises(ValueError):
            model2.build(input_var)

    def test_same_model_with_no_name(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        model.build(input_var)
        with self.assertRaises(ValueError):
            model.build(input_var)

    def test_model_with_different_name(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        model.build(input_var)
        model.build(input_var, name='model_2')
        data = np.ones((3, 5))
        results, results2 = self.sess.run(
            [model.outputs, model.model_2.outputs],
            feed_dict={input_var: data})
        assert np.array_equal(results, results2)

    def test_model_with_different_name_in_different_order(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        model.build(input_var, name='model_2')
        model.build(input_var)
        data = np.ones((3, 5))
        results, results2 = self.sess.run(
            [model.outputs, model.model_2.outputs],
            feed_dict={input_var: data})
        assert np.array_equal(results, results2)

    def test_model_child_not_pickable(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        model.build(input_var, name='child')
        with self.assertRaises(TypeError):
            pickle.loads(pickle.dumps(model.child))

    def test_model_pickle(self):
        data = np.ones((3, 5))
        model = SimpleModel(output_dim=2)
        
        with tf.Session(graph=tf.Graph()) as sess:
            input_var = tf.placeholder(tf.float32, shape=(None, 5))
            model.build(input_var)
            
            results = sess.run(model.outputs, feed_dict={input_var: data})
            model_data = pickle.dumps(model)

        with tf.Session(graph=tf.Graph()) as sess:
            input_var = tf.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(model_data)
            model_pickled.build(input_var)

            results2 = sess.run(
                model_pickled.outputs, feed_dict={input_var: data})

        assert np.array_equal(results, results2)

    def test_model_pickle_same_parameters(self):
        model = SimpleModel(output_dim=2)

        with tf.Session(graph=tf.Graph()) as sess:
            state = tf.placeholder(shape=[None, 10, 5], dtype=tf.float32)
            model.build(state)

            model.parameters = {
                k: np.zeros_like(v)
                for k, v in model.parameters.items()
            }
            all_one = {k: np.ones_like(v) for k, v in model.parameters.items()}
            model.parameters = all_one
            h_data = pickle.dumps(model)

        with tf.Session(graph=tf.Graph()) as sess:
            model_pickled = pickle.loads(h_data)
            state = tf.placeholder(shape=[None, 10, 5], dtype=tf.float32)
            model_pickled.build(state)

            np.testing.assert_equal(all_one, model_pickled.parameters)
