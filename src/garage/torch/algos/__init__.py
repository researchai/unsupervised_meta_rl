"""PyTorch algorithms."""
from garage.torch.algos.ddpg import DDPG
from garage.torch.algos.pearl_inference_network import PEARLInferenceNetwork
from garage.torch.algos.ppo import PPO  # noqa: I100
from garage.torch.algos.recurrent_encoder import RecurrentEncoder
from garage.torch.algos.trpo import TRPO
from garage.torch.algos.vpg import VPG

__all__ = [
    'DDPG', 'VPG', 'PPO', 'TRPO', 'PEARLInferenceNetwork', 'RecurrentEncoder'
]
