class Algorithm:
    pass


class RLAlgorithm(Algorithm):
    def get_actions(self, obs):
        r"""Get actions given a batch of observations.

        Args:
            obs: ndarray(batch_size, obs_dim)
                 Observations.

        Returns: ndarray(batch_size, action_dim)
                 Actions under current policy.

        """
        raise NotImplementedError

    def train_once(self, paths):
        r"""Update policy given one batch of samples.

        Args:
            paths: [Path]
                Path: {
                    observations: ndarray(path_len, obs_dim),
                    actions: ndarray(path_len, action_dim),
                    rewards: ndarray(path_len),
                    infos: array(path_len)
                }

        """
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
