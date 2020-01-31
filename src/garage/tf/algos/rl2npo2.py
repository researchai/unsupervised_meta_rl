"""Natural Policy Gradient Optimization."""
from dowel import logger
import numpy as np
import copy

from garage.misc import tensor_utils as np_tensor_utils
from garage.tf.algos import NPO
from garage.tf.misc.tensor_utils import compute_advantages_individual
from garage.tf.misc.tensor_utils import discounted_returns_individual
from garage.tf.misc.tensor_utils import split_paths


class RL2NPO2(NPO):
    """Natural Policy Gradient Optimization used in RL2."""
    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 meta_batch_size,
                 episode_per_task,
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
        self._episode_per_task = episode_per_task
        self._baselines = [copy.deepcopy(baseline) for _ in range(meta_batch_size)]
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

    def _fit_baseline_first(self, samples_data):
        """Update baselines from samples and get baseline prediction.

        Args:
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        """
        # Get baseline prediction
        all_paths = samples_data['paths']
        lengths = samples_data['lengths']
        baselines = np.zeros((self._meta_batch_size, self.max_path_length))
        for ind, paths in enumerate(all_paths):
            split_path = split_paths(paths, lengths[ind])
            baselines[ind] = np.concatenate([self._baselines[ind].predict(path) for path in split_path])

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

        paths = samples_data['paths']
        valids = samples_data['valids']
        lengths = samples_data['lengths']
        # Fit baseline
        logger.log('Fitting baseline...')
        for ind, path in enumerate(paths):
            path['rewards'] = rewards_tensor[ind][valids[ind].astype(np.bool)]
            path['returns'] = returns_tensor[ind][valids[ind].astype(np.bool)]
            split_path = split_paths(path, lengths[ind])
            self._baselines[ind].fit(split_path)

    def _get_advantages(self, baseline_var, reward_var, name):
        return compute_advantages_individual(self.discount,
                                             self.gae_lambda,
                                             self.max_path_length,
                                             self._episode_per_task,
                                             baseline_var,
                                             reward_var,
                                             name=name)

    def _get_discounted_returns(self, rewards):
        return discounted_returns_individual(self.discount,
                                             self.max_path_length,
                                             self._episode_per_task,
                                             rewards)