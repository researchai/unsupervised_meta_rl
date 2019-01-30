"""Parameter layer in TensorFlow."""
import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer


class DistributionLayer(KerasLayer):
    def __init__(self, dist, seed, **kwargs):
        super().__init__(**kwargs)
        self._dist_callable = dist
        self._seed = seed

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, input):
        self._dist = self._dist_callable(*input)
        return self._dist.sample(seed=self._seed)

    def get_config(self):
        """Cusomterized configuration for serialization."""
        config = {
            "dist": self._dist_callable,
            "seed": self._seed,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
