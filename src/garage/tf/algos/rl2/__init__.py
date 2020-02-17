"""Module for RL2."""
from garage.tf.algos.rl2.rl2 import RL2
from garage.tf.algos.rl2.rl2env import RL2Env
from garage.tf.algos.rl2.rl2npo import RL2NPO
from garage.tf.algos.rl2.rl2ppo import RL2PPO

__all__ = ['RL2', 'RL2Env', 'RL2NPO', 'RL2PPO']
