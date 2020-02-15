"""Natural Policy Gradient Optimization."""
from dowel import logger
import numpy as np

from garage.misc import tensor_utils as np_tensor_utils
from garage.tf.algos import NPO


class RL2NPO(NPO):
    """Natural Policy Gradient Optimization used in RL2."""

    def _fit_baseline_first(self, samples_data):
        """Update baselines from samples and get baseline prediction.

        Args:
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        """
        # policy_opt_input_values = self._policy_opt_input_values(samples_data)

        # # Augment reward from baselines
        # rewards_tensor = self._f_rewards(*policy_opt_input_values)
        # returns_tensor = self._f_returns(*policy_opt_input_values)
        # returns_tensor = np.squeeze(returns_tensor, -1)
        
        paths = samples_data['paths']
        # valids = samples_data['valids']

        # Fit baseline
        logger.log('Fitting baseline...')
        # for ind, path in enumerate(paths):
        #     path['rewards'] = rewards_tensor[ind][valids[ind].astype(np.bool)]
        #     path['returns'] = returns_tensor[ind][valids[ind].astype(np.bool)]

        self.baseline.fit(paths)
        baselines = [self.baseline.predict(path) for path in paths]
            # self.baseline.fit([path])
            # baseline = self.baseline.predict(path)
            # baselines.append(baseline)

        baselines = np_tensor_utils.pad_tensor_n(baselines, self.max_path_length)
        samples_data['baselines'] = baselines

    def _fit_baseline_after(self, samples_data):
        """Update baselines from samples.
        Args:
            samples_data (dict): Processed sample data.
                See process_samples() for details.
        """
        pass