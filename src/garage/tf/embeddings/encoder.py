"""Encoders in TensorFlow."""
# pylint: disable=abstract-method
from garage.np.embeddings import Encoder as BaseEncoder
from garage.np.embeddings import StochasticEncoder as BaseStochasticEncoder
from garage.tf.models import Module, StochasticModule


class Encoder(BaseEncoder, Module):
    """Base class for encoders in TensorFlow."""

    def forward_n(self, input_values):
        """Get samples of embedding for the given inputs.

        Args:
            input_values (numpy.ndarray): Tensors to encode.

        Returns:
            numpy.ndarray: Embeddings sampled from embedding distribution.
            dict: Embedding distribution information.

        Note:
            It returns an embedding and a dict, with keys
            - mean (list[numpy.ndarray]): Means of the distribution.
            - log_std (list[numpy.ndarray]): Log standard deviations of the
                distribution.

        """


class StochasticEncoder(BaseStochasticEncoder, StochasticModule):
    """Base class for stochastic encoders in TensorFlow."""

    def build(self, embedding_input, name=None):
        """Build encoder.

        After buil, self.distribution is a Gaussian distribution conitioned
        on embedding_input.

        Args:
          embedding_input (tf.Tensor) : Embedding input.
          name (str): Name of the model, which is also the name scope.

        """
