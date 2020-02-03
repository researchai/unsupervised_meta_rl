"""Natural Policy Gradient Optimization."""
from dowel import logger
import numpy as np
import copy

from garage.misc import tensor_utils as np_tensor_utils
from garage.tf.algos import NPO


class RL2NPO2(NPO):
    """Natural Policy Gradient Optimization used in RL2."""
    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 meta_batch_size,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate',
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
                 center_adv_across_batch=True,
                 name='NPO'):
        self._meta_batch_size = meta_batch_size
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
                         center_adv_across_batch=center_adv_across_batch,
                         flatten_input=flatten_input)
        if baseline.recurrent:
            self._baselines = [
                type(baseline)(
                    env_spec=baseline._mdp_spec,
                    regressor_args=baseline._regressor_args,
                    name="{}-{}".format(baseline.name, i))
                for i in range(meta_batch_size)
            ]
        else:
            self._baselines = [
                type(baseline)(
                    env_spec=baseline._mdp_spec,
                    name="{}-{}".format(baseline.name, i))
                for i in range(meta_batch_size)
            ] 

    def _fit_baseline_first(self, samples_data):
        """Update baselines from samples and get baseline prediction.

        Args:
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        """
        # Get baseline prediction
        paths = samples_data['paths']
        baselines = []
        for ind, path in enumerate(paths):
            baseline = self._baselines[ind].predict(path)
            baselines.append(baseline)

        baselines = np_tensor_utils.pad_tensor_n(baselines, self.max_path_length)
        samples_data['baselines'] = baselines

    def _fit_baseline_after(self, samples_data):
        """Update baselines from samples.
        Args:
            samples_data (dict): Processed sample data.
                See process_samples() for details.
        """
        policy_opt_input_values = self._policy_opt_input_values(samples_data)

        # Augment reward from baselines
        rewards_tensor = self._f_rewards(*policy_opt_input_values)
        returns_tensor = self._f_returns(*policy_opt_input_values)
        returns_tensor = np.squeeze(returns_tensor, -1)
        adv = self._f_adv(*policy_opt_input_values)

        paths = samples_data['paths']
        valids = samples_data['valids']
        # Fit baseline
        logger.log('Fitting baseline...')
        for ind, path in enumerate(paths):
            path['rewards'] = rewards_tensor[ind][valids[ind].astype(np.bool)]
            path['returns'] = returns_tensor[ind][valids[ind].astype(np.bool)]
            self._baselines[ind].fit([path])
