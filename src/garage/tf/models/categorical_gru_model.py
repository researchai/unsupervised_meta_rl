"""GRU Model.

A model composed only of a Gated Recurrent Unit (GRU).
"""
import tensorflow as tf
import tensorflow_probability as tfp

from garage.tf.models.gru_model2 import GRUModel2


class CategoricalGRUModel(GRUModel2):
    """Categorical GRU Model.

    Args:
        output_dim (int): Dimension of the network output.
        hidden_dim (int): Hidden dimension for GRU cell.
        name (str): Policy name, also the variable scope.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        recurrent_nonlinearity (callable): Activation function for recurrent
            layers. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        recurrent_w_init (callable): Initializer function for the weight
            of recurrent layer(s). The function should return a
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
        hidden_state_init (callable): Initializer function for the
            initial hidden state. The functino should return a tf.Tensor.
        hidden_state_init_trainable (bool): Bool for whether the initial
            hidden state is trainable.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 output_dim,
                 hidden_dim,
                 name=None,
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 recurrent_nonlinearity=tf.nn.sigmoid,
                 recurrent_w_init=tf.glorot_uniform_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 hidden_state_init=tf.zeros_initializer(),
                 hidden_state_init_trainable=False,
                 layer_normalization=False):
        super().__init__(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            name=name,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            recurrent_nonlinearity=recurrent_nonlinearity,
            recurrent_w_init=recurrent_w_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            hidden_state_init=hidden_state_init,
            hidden_state_init_trainable=hidden_state_init_trainable,
            layer_normalization=layer_normalization)

    def network_output_spec(self):
        """Network output spec."""
        return ['all_output', 'step_output', 'step_hidden', 'init_hidden', 'dist']

    def _build(self, state_input, step_input, step_hidden, name=None):
        prob, step_output, step_hidden, init_hidden = super()._build(
            state_input=state_input,
            step_input_var=step_input,
            step_hidden_var=step_hidden,
            name=name)
        dist = tfp.distributions.OneHotCategorical(prob)
        return prob, step_output, step_hidden, init_hidden, dist
