"""Base class for batch sampling-based policy optimization methods."""
from abc import abstractmethod
import collections

from dowel import logger

from garage.np.algos import RLAlgorithm
from garage.sampler import MultiprocessingSampler


class BatchPolopt(RLAlgorithm):
    """Base class for batch sampling-based policy optimization methods.

    This includes various policy gradient methods like VPG, NPG, PPO, TRPO,
    etc.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        flatten_input (bool): Whether to flatten input along the observation
            dimension. If True, for example, an observation with shape (2, 4)
            will be flattened to 8.

    """

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
                 flatten_input=True):
        self.env_spec = env_spec
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.fixed_horizon = fixed_horizon
        self.flatten_input = flatten_input

        self.episode_reward_mean = collections.deque(maxlen=100)

        self.sampler_cls = MultiprocessingSampler

        self.init_opt()

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            runner.step_path = runner.obtain_samples(runner.step_itr)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Average return.

        """
        raise NotImplementedError

    def log_diagnostics(self, paths):
        """Log diagnostic information.

        Args:
            paths (list[dict]): A list of collected paths.

        """
        logger.log('Logging diagnostics...')
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """Initialize the optimization procedure.

        If using tensorflow, this may include declaring all the variables and
        compiling functions.
        """
        raise NotImplementedError

    @abstractmethod
    def optimize_policy(self, itr, samples_data):
        """Optimize the policy using the samples.

        Args:
            itr (int): Iteration number.
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        Raises:
            NotImplementedError: Raise when child class
                does not overwrite this method.

        """
        del itr
        del samples_data
        raise NotImplementedError
