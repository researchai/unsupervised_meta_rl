import abc


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_actions(self, obs):
        """
        Get actions given a batch of observations.

        Args:
            obs: Tensor(batch_size, observation_dim)
                 Observations.

        Returns: Tensor(batch_size, action_dim)
                 Actions.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_once(self, samples):
        """
        Train policy given trajectories sampled under latest policy.

        Args:
            samples: [Trajectory]
                Trajectory: {
                    observations: Tensor(path_len, observation_dim),
                    actions: Tensor(path_len, action_dim),
                    rewards: Tensor(path_len)
                }

        """
        raise NotImplementedError
