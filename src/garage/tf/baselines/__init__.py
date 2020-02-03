"""Baseline estimators for TensorFlow-based algorithms."""
from garage.tf.baselines.continuous_mlp_baseline import ContinuousMLPBaseline
from garage.tf.baselines.gaussian_cnn_baseline import GaussianCNNBaseline
from garage.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from garage.tf.baselines.gaussian_gru_baseline import GaussianGRUBaseline

__all__ = [
    'ContinuousMLPBaseline',
    'GaussianCNNBaseline',
    'GaussianGRUBaseline',
    'GaussianMLPBaseline',
]
