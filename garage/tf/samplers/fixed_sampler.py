#import numpy as np
#import tensorflow as tf

from garage.sampler import parallel_sampler
from garage.tf.samplers import BatchSampler
from garage.misc.overrides import overrides


class FixedSampler(BatchSampler):
    """
    Assume self.algo contains attribute expert_paths or expert_policy
    :type self.algo.expert_policy Policy
    :type self.algo.expert_paths list(paths?)
    """

    @overrides
    def obtain_samples(self, itr):
        """
        Sample expert policy for a set of paths once, if expert policy defined
        Otherwise assume expert paths already defined
        On subsequent calls, return the same set of paths
        """
        if self.algo.expert_paths is not None:
            return self.algo.expert_paths

        if self.algo.expert_policy is None:
            return None  # How to handle errors in garage?

        # TODO: How do we ensure seed is set to a fixed value?
        # parallel_sampler.set_seed(42)

        cur_policy_params = self.algo.expert_policy.get_param_values()
        cur_env_params = self.algo.env.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_policy_params,
            env_params=cur_env_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(
                paths, self.algo.batch_size)
            return paths_truncated
