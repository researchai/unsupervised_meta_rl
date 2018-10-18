import tensorflow as tf

from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.algos import BatchPolopt
from garage.tf.misc import tensor_utils
from garage.tf.samplers import FixedSampler
from garage.tf.misc.tensor_utils import flatten_inputs
from garage.tf.misc.tensor_utils import graph_inputs
from garage.tf.optimizers import LbfgsOptimizer


class BC(BatchPolopt):
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 sampler_cls=None,
                 sampler_args=None,
                 name="BC",
                 policy=None,
                 **kwargs):
        self.name = name
        self._name_scope = tf.name_scope(self.name)

        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = LbfgsOptimizer

        if sampler_cls is None:
            if sampler_args is None:
                sampler_args = dict()
            sampler_cls = FixedSampler

        with self._name_scope:
            self.optimizer = optimizer(**optimizer_args)

        super(BC, self).__init__(policy=policy, **kwargs)

    @overrides
    def init_opt(self):
        pol_loss_inputs, pol_opt_inputs = self._build_inputs()
        self._policy_opt_inputs = pol_opt_inputs

        pol_loss = self._build_policy_loss(pol_loss_inputs)
        self.optimizer.update_opt(
            loss=pol_loss,
            target=self.policy,
            inputs=flatten_inputs(self._policy_opt_inputs))

        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        policy_opt_input_values = self._policy_opt_input_values(samples_data)

        loss_before = self.optimizer.loss(policy_opt_input_values)
        self.optimizer.optimize(policy_opt_input_values)
        loss_after = self.optimizer.loss(policy_opt_input_values)

        logger.record_tabular("{}/LossBefore".format(self.policy.name),
                              loss_before)
        logger.record_tabular("{}/LossAfter".format(self.policy.name),
                              loss_after)
        logger.record_tabular("{}/dLoss".format(self.policy.name),
                              loss_before - loss_after)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

    def _build_inputs(self):
        observation_space = self.policy.observation_space
        action_space = self.policy.action_space

        policy_dist = self.policy.distribution

        with tf.name_scope("inputs"):
            obs_var = observation_space.new_tensor_variable(
                name="obs", extra_dims=2)
            action_var = action_space.new_tensor_variable(
                name="action", extra_dims=2)
            reward_var = tensor_utils.new_tensor(
                name="reward", ndim=2, dtype=tf.float32)

            policy_state_info_vars = {
                k: tf.placeholder(
                    tf.float32, shape=[None] * 2 + list(shape), name=k)
                for k, shape in self.policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.state_info_keys
            ]

            # old policy distribution
            policy_old_dist_info_vars = {
                k: tf.placeholder(
                    tf.float32,
                    shape=[None] * 2 + list(shape),
                    name="policy_old_%s" % k)
                for k, shape in policy_dist.dist_info_specs
            }
            policy_old_dist_info_vars_list = [
                policy_old_dist_info_vars[k]
                for k in policy_dist.dist_info_keys
            ]

            policy_loss_inputs = graph_inputs(
                "PolicyLossInputs",
                obs_var=obs_var,
                action_var=action_var,
                reward_var=reward_var,
                policy_state_info_vars=policy_state_info_vars,
                policy_old_dist_info_vars=policy_old_dist_info_vars,
            )
            policy_opt_inputs = graph_inputs(
                "PolicyOptInputs",
                obs_var=obs_var,
                action_var=action_var,
                reward_var=reward_var,
                policy_state_info_vars_list=policy_state_info_vars_list,
                policy_old_dist_info_vars_list=policy_old_dist_info_vars_list,
            )

        return policy_loss_inputs, policy_opt_inputs

    def _build_policy_loss(self, loss_inputs):
        pol_dist = self.policy.distribution

        with tf.name_scope("policy_loss"):
            pol_loss = None

        return pol_loss

    def _policy_opt_input_values(self, samples_data):
        """ Map rollout samples to the policy optimizer inputs """
        policy_state_info_list = [
            samples_data["agent_infos"][k] for k in self.policy.state_info_keys
        ]
        policy_old_dist_info_list = [
            samples_data["agent_infos"][k]
            for k in self.policy.distribution.dist_info_keys
        ]

        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=samples_data["observations"],
            action_var=samples_data["actions"],
            reward_var=samples_data["rewards"],
            policy_state_info_vars_list=policy_state_info_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_list,
        )

        return flatten_inputs(policy_opt_input_values)
