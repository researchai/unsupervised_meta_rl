"""Natural Policy Gradient Optimization."""
from dowel import logger, tabular
import numpy as np
import tensorflow as tf

from garage.misc import tensor_utils as np_tensor_utils
from garage.tf.algos import RL2NPO
from garage.tf.optimizers import FirstOrderOptimizer


class RL2PPO(RL2NPO):
    """Natural Policy Gradient Optimization used in RL2."""
    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate_clip',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 flatten_input=True,
                 name='PPO'):
        if optimizer is None:
            optimizer = FirstOrderOptimizer
            if optimizer_args is None:
                optimizer_args = dict()
        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         scope=scope,
                         max_path_length=max_path_length,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         fixed_horizon=fixed_horizon,
                         pg_loss=pg_loss,
                         lr_clip_range=lr_clip_range,
                         max_kl_step=max_kl_step,
                         optimizer=optimizer,
                         optimizer_args=optimizer_args,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         use_neg_logli_entropy=use_neg_logli_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         flatten_input=flatten_input,
                         name=name)