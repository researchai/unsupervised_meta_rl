"""Exploration strategies which use NumPy as a numerical backend."""
from garage.np.exploration_strategies.base import ExplorationStrategy
from garage.np.exploration_strategies.epsilon_greedy_strategy import (
    EpsilonGreedyStrategy)
from garage.np.exploration_strategies.ou_strategy import OUStrategy
from garage.np.exploration_strategies.gaussian_strategy import GaussianStrategy

__all__ = ['EpsilonGreedyStrategy', 'ExplorationStrategy', 'OUStrategy', 'GaussianStrategy']
