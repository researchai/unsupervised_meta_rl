
from enum import Enum
from enum import unique
import time

import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.misc import special
from garage.misc.overrides import overrides
from garage.tf.algos import BatchPolopt
from garage.tf.algos.npo import PGLoss
from garage.tf.misc import tensor_utils
from garage.tf.misc.tensor_utils import compute_advantages
from garage.tf.misc.tensor_utils import discounted_returns
from garage.tf.misc.tensor_utils import filter_valids
from garage.tf.misc.tensor_utils import filter_valids_dict
from garage.tf.misc.tensor_utils import flatten_batch
from garage.tf.misc.tensor_utils import flatten_batch_dict
from garage.tf.misc.tensor_utils import flatten_inputs
from garage.tf.misc.tensor_utils import graph_inputs
from garage.tf.optimizers import LbfgsOptimizer
from garage.tf.policies.maml_policy import MamlPolicy
from garage.tf.samplers.multitask_vectorized_sampler import MultitaskVecterizedSampler


class MAML(BatchPolopt):

    """
    Model-Agnostic Meta Learning with policy gradient.
    """

    def __init__(self,
                 pg_loss=PGLoss.VANILLA,
                 clip_range=0.01,
                 optimizer=None,
                 optimizer_args=dict(),
                 name="MAML",
                 policy=None,
                 policy_ent_coeff=0.0,
                 sampler_cls=MultitaskVecterizedSampler,
                 **kwargs):

        assert isinstance(policy, MamlPolicy)
        self.name = name

        self.pg_loss = pg_loss

        # Optimizer
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = LbfgsOptimizer
        with tf.name_scope(self.name):
            self.optimizer = optimizer(**optimizer_args)
            self.clip_range = float(clip_range)
            self.policy_ent_coeff = float(policy_ent_coeff)
        super(MAML, self).__init__(
            policy=policy,
            sampler_cls=sampler_cls,
            **kwargs)

    def init_opt(self):
        # Build inputs
        loss_inputs, opt_inputs, kl_inputs, kl_opt_inputs = self._build_inputs()
        self._policy_loss_inputs = loss_inputs
        self._policy_opt_inputs = opt_inputs
        self._kl_inputs = kl_inputs
        self._kl_opt_inputs = kl_opt_inputs  # TODO: Rename this..

        # Compute the loss for one step adaptation and gradients
        adaptation_losses = self._build_adaption_loss(loss_inputs)
        gradient_vars = self._build_gradient_vars(adaptation_losses)

        # Build one step adaptation operations.
        obs_flat_vars = [i.flat.obs_var for i in loss_inputs]
        maml_infos, adapt_opt, adapt_opt_input = self.policy.initialize(gradient_vars, inputs=obs_flat_vars)

        self.f_adapt = tensor_utils.compile_function(
            flatten_inputs(opt_inputs[0]),
            adapt_opt,
            log_name="adaptation",
        )

        # Buil inputs to compute the final loss
        pol_loss, pol_kl = self._build_final_loss(loss_inputs, kl_inputs, maml_infos)
        self.optimizer.update_opt(
            loss=pol_loss,
            target=self.policy.wrapped_policy,
            leq_constraint=(pol_kl, self.clip_range),
            inputs=flatten_inputs([self._policy_opt_inputs, self._kl_opt_inputs]),
            constraint_name="mean_kl")

        print("done with initializing the whole graph!")

    def _build_inputs(self):
        loss_inputs = []
        opt_inputs = []

        observation_space = self.policy.wrapped_policy.observation_space
        action_space = self.policy.wrapped_policy.action_space
        policy_dist = self.policy.wrapped_policy.distribution
        
        for i in range(self.policy.n_tasks):
            with tf.name_scope("inputs_{}".format(i)):
                obs_var = observation_space.new_tensor_variable(
                    name="obs", extra_dims=2)
                action_var = action_space.new_tensor_variable(
                    name="action", extra_dims=2)
                reward_var = tensor_utils.new_tensor(
                    name="reward", ndim=2, dtype=tf.float32)
                valid_var = tf.placeholder(
                    tf.float32, shape=[None, None], name="valid")
                baseline_var = tensor_utils.new_tensor(
                    name="baseline", ndim=2, dtype=tf.float32)

                policy_state_info_vars = {
                    k: tf.placeholder(
                        tf.float32, shape=[None] * 2 + list(shape), name=k)
                    for k, shape in self.policy.wrapped_policy.state_info_specs
                }
                policy_state_info_vars_list = [
                    policy_state_info_vars[k] for k in self.policy.wrapped_policy.state_info_keys
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

                # flattened view
                with tf.name_scope("flat"):
                    obs_flat = flatten_batch(obs_var, name="obs_flat")
                    action_flat = flatten_batch(action_var, name="action_flat")
                    reward_flat = flatten_batch(reward_var, name="reward_flat")
                    valid_flat = flatten_batch(valid_var, name="valid_flat")
                    policy_state_info_vars_flat = flatten_batch_dict(
                        policy_state_info_vars, name="policy_state_info_vars_flat")
                    policy_old_dist_info_vars_flat = flatten_batch_dict(
                        policy_old_dist_info_vars,
                        name="policy_old_dist_info_vars_flat")

                # valid view
                with tf.name_scope("valid"):
                    action_valid = filter_valids(
                        action_flat, valid_flat, name="action_valid")
                    policy_state_info_vars_valid = filter_valids_dict(
                        policy_state_info_vars_flat,
                        valid_flat,
                        name="policy_state_info_vars_valid")
                    policy_old_dist_info_vars_valid = filter_valids_dict(
                        policy_old_dist_info_vars_flat,
                        valid_flat,
                        name="policy_old_dist_info_vars_valid")
            
            # policy loss and optimizer inputs
            pol_flat = graph_inputs(
                "PolicyLossInputsFlat",
                obs_var=obs_flat,
                action_var=action_flat,
                reward_var=reward_flat,
                valid_var=valid_flat,
                policy_state_info_vars=policy_state_info_vars_flat,
                policy_old_dist_info_vars=policy_old_dist_info_vars_flat,
            )
            pol_valid = graph_inputs(
                "PolicyLossInputsValid",
                action_var=action_valid,
                policy_state_info_vars=policy_state_info_vars_valid,
                policy_old_dist_info_vars=policy_old_dist_info_vars_valid,
            )
            policy_loss_inputs = graph_inputs(
                "PolicyLossInputs",
                obs_var=obs_var,
                action_var=action_var,
                reward_var=reward_var,
                baseline_var=baseline_var,
                valid_var=valid_var,
                policy_state_info_vars=policy_state_info_vars,
                policy_old_dist_info_vars=policy_old_dist_info_vars,
                flat=pol_flat,
                valid=pol_valid,
            )
            policy_opt_inputs = graph_inputs(
                "PolicyOptInputs",
                obs_var=obs_var,
                action_var=action_var,
                reward_var=reward_var,
                baseline_var=baseline_var,
                valid_var=valid_var,
                policy_state_info_vars_list=policy_state_info_vars_list,
                policy_old_dist_info_vars_list=policy_old_dist_info_vars_list,
            )
            loss_inputs.append(policy_loss_inputs)
            opt_inputs.append(policy_opt_inputs)

        # Build an extra obs_var for computing the KL divergence
        # because we compute the kl constraint over the whole batch.
        with tf.name_scope("kl_inputs"):
            obs_var = observation_space.new_tensor_variable(name="obs", extra_dims=2)
            obs_var_flat = flatten_batch(obs_var, name="obs_flat")
            valid_var = tf.placeholder(
                tf.float32, shape=[None, None], name="valid")
            valid_flat = flatten_batch(valid_var, name="valid_flat")
            policy_old_dist_info_vars = {
                k: tf.placeholder(
                    tf.float32,
                    shape=[None] * 2 + list(shape),
                    name="policy_old_%s" % k)
                for k, shape in policy_dist.dist_info_specs
            }
            policy_state_info_vars = {
                k: tf.placeholder(
                    tf.float32, shape=[None] * 2 + list(shape), name=k)
                for k, shape in self.policy.wrapped_policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.wrapped_policy.state_info_keys
            ]
            policy_old_dist_info_vars_list = [
                policy_old_dist_info_vars[k]
                for k in policy_dist.dist_info_keys
            ]
            
            policy_old_dist_info_vars_flat = flatten_batch_dict(
                policy_old_dist_info_vars,
                name="policy_old_dist_info_vars_flat")
            policy_state_info_vars_flat = flatten_batch_dict(
                policy_state_info_vars, name="policy_state_info_vars_flat")
            
            policy_old_dist_info_vars_valid = filter_valids_dict(
                policy_old_dist_info_vars_flat,
                valid_flat,
                name="policy_old_dist_info_vars_valid")
            policy_state_info_vars_valid = filter_valids_dict(
                policy_state_info_vars_flat,
                valid_flat,
                name="policy_state_info_vars_valid")

            kl_inputs_flat = graph_inputs(
                "KLInputsFlat",
                obs_var=obs_var_flat,
                policy_old_dist_info_vars=policy_old_dist_info_vars_flat,
                valid_var=valid_flat,
                policy_state_info_vars=policy_state_info_vars_flat,
            )
            kl_inputs_valid = graph_inputs(
                "KLInputsValid",
                policy_old_dist_info_vars=policy_old_dist_info_vars_valid,
                policy_state_info_vars=policy_old_dist_info_vars_valid
            )
            kl_inputs = graph_inputs(
                "KLInputs",
                flat=kl_inputs_flat,
                valid=kl_inputs_valid,
                obs_var=obs_var,
                valid_var=valid_var,
                policy_old_dist_info_vars=policy_old_dist_info_vars,
                policy_state_info_vars=policy_state_info_vars,
            )
            kl_opt_inputs = graph_inputs(
                "KLOptInputs",
                obs_var=obs_var,
                valid_var=valid_var,
                policy_old_dist_info_vars_list=policy_old_dist_info_vars_list,
                policy_state_info_vars_list=policy_state_info_vars_list,
            )

        return loss_inputs, opt_inputs, kl_inputs, kl_opt_inputs

    def _build_adaption_loss(self, inputs):
        self.f_rewards = []
        self.f_returns = []
        losses = []
        pol_dist = self.policy.wrapped_policy.distribution
        for idx, i in enumerate(inputs):
            with tf.name_scope("adaptation_loss_{}".format(idx)):
                policy_entropy = self._build_entropy_term(i)
                with tf.name_scope("augmented_rewards"):
                    rewards = i.reward_var + self.policy_ent_coeff * policy_entropy

                with tf.name_scope("policy_loss"):
                    advantages = compute_advantages(
                        self.discount,
                        self.gae_lambda,
                        self.max_path_length,
                        i.baseline_var,
                        rewards,
                        name="advantages")
                    
                    adv_flat = flatten_batch(advantages, name="adv_flat")
                    adv_valid = filter_valids(
                        adv_flat, i.flat.valid_var, name="adv_valid")

                    if self.policy.recurrent:
                        advantages = tf.reshape(advantages, [-1, self.max_path_length])

                    # Optionally normalize advantages
                    eps = tf.constant(1e-8, dtype=tf.float32)
                    if self.center_adv:
                        with tf.name_scope("center_adv"):
                            mean, var = tf.nn.moments(adv_valid, axes=[0])
                            adv_valid = tf.nn.batch_normalization(
                                adv_valid, mean, var, 0, 1, eps)
                    if self.positive_adv:
                        with tf.name_scope("positive_adv"):
                            m = tf.reduce_min(adv_valid)
                            adv_valid = (adv_valid - m) + eps

                    if self.policy.recurrent:
                        policy_dist_info = self.policy.wrapped_policy.dist_info_sym(
                            i.obs_var,
                            i.policy_state_info_vars,
                            name="policy_dist_info")
                    else:
                        policy_dist_info_flat = self.policy.wrapped_policy.dist_info_sym(
                            i.flat.obs_var,
                            i.flat.policy_state_info_vars,
                            name="policy_dist_info_flat")

                        policy_dist_info_valid = filter_valids_dict(
                            policy_dist_info_flat,
                            i.flat.valid_var,
                            name="policy_dist_info_valid")

                    # Calculate surrogate loss
                    with tf.name_scope("surr_loss"):
                        if self.policy.recurrent:
                            lr = pol_dist.likelihood_ratio_sym(
                                i.action_var,
                                i.policy_old_dist_info_vars,
                                policy_dist_info,
                                name="lr")

                            surr_vanilla = lr * advantages * i.valid_var
                        else:
                            lr = pol_dist.likelihood_ratio_sym(
                                i.valid.action_var,
                                i.valid.policy_old_dist_info_vars,
                                policy_dist_info_valid,
                                name="lr")

                            surr_vanilla = lr * adv_valid

                        # Always use vanilla surrogate loss for adaptation
                        surr_obj = tf.identity(surr_vanilla, name="surr_obj")

                        # Maximize E[surrogate objective] by minimizing
                        # -E_t[surrogate objective]
                        if self.policy.recurrent:
                            surr_loss = (-tf.reduce_sum(surr_obj)) / tf.reduce_sum(
                                i.valid_var)
                        else:
                            surr_loss = -tf.reduce_mean(surr_obj)
                    
                    self.f_rewards.append(tensor_utils.compile_function(
                        flatten_inputs(self._policy_opt_inputs),
                        rewards,
                        log_name="f_rewards"))
                    returns = discounted_returns(self.discount, self.max_path_length,
                                         rewards)
                    self.f_returns.append(tensor_utils.compile_function(
                        flatten_inputs(self._policy_opt_inputs),
                        returns,
                        log_name="f_returns"))
                    losses.append(surr_loss)
        return losses

    def _build_gradient_vars(self, losses):
        params = self.policy.wrapped_policy.get_params_internal()
        gradient_vars = []
        for idx, l in enumerate(losses):
            with tf.name_scope("adaptation_gradients_{}".format(idx)):
                grads = [tf.gradients(l, p)[0] for p in params]
                gradient_vars.append(grads)

        return gradient_vars

    def _build_entropy_term(self, i):
        with tf.name_scope("policy_entropy"):
            if self.policy.wrapped_policy.recurrent:
                policy_dist_info_flat = self.policy.wrapped_policy.dist_info_sym(
                    i.obs_var,
                    i.policy_state_info_vars,
                    name="policy_dist_info")
                policy_neg_log_likeli_flat = self.policy.wrapped_policy.distribution.log_likelihood_sym(  # noqa: E501
                    i.action_var,
                    policy_dist_info_flat,
                    name="policy_log_likeli")
            else:
                policy_dist_info_flat = self.policy.wrapped_policy.dist_info_sym(
                    i.flat.obs_var,
                    i.flat.policy_state_info_vars,
                    name="policy_dist_info_flat")
                policy_neg_log_likeli_flat = self.policy.wrapped_policy.distribution.log_likelihood_sym(  # noqa: E501
                    i.flat.action_var,
                    policy_dist_info_flat,
                    name="policy_log_likeli")


            policy_entropy_flat = self.policy.wrapped_policy.distribution.entropy_sym(
                policy_dist_info_flat)

            policy_entropy = tf.reshape(policy_entropy_flat,
                                        [-1, self.max_path_length])

            policy_entropy = policy_entropy * i.valid_var
            policy_entropy = tf.stop_gradient(policy_entropy)

        return policy_entropy

    def _build_final_loss(self, loss_inputs, kl_inputs, maml_infos):
        # Build the entropy term first.
        # At this point the maml policy is not nicely constructed
        # so just use the output terms build everything.
        # TODO: add recurrent version
        pol_dist = self.policy.wrapped_policy.distribution

        dist_infos_flat = []
        for idx in range(self.policy.n_tasks):
            outputs = maml_infos[idx][1]
            dist_infos_flat.append(dict(mean=outputs["mean"], log_std=outputs["std_param"]))
        entropy_syms = [
            pol_dist.entropy_sym(info)
            for info in dist_infos_flat
        ]
        policy_entropies = [
            tf.reshape(ent, [-1, self.max_path_length])
            for ent in entropy_syms
        ]
        
        # filter entropies with varlid vars
        policy_entropies_valid = [
            ent * i.valid_var
            for ent, i in zip(policy_entropies, loss_inputs)
        ]

        losses = []
        # Build a loss for each of the adapted policies
        for idx, (i, infos) in enumerate(zip(loss_inputs, maml_infos)):
            with tf.name_scope("adapted_{}".format(idx)):
                # Augmented rewards
                with tf.name_scope("augmented_rewards"):
                    rewards = i.reward_var\
                        + (self.policy_ent_coeff * policy_entropies_valid[idx])
                # Loss
                with tf.name_scope("policy_loss"):
                    advantages = compute_advantages(
                        self.discount,
                        self.gae_lambda,
                        self.max_path_length,
                        i.baseline_var,
                        rewards,
                        name="advantages")
                    adv_flat = flatten_batch(advantages, name="adv_flat")
                    adv_valid = filter_valids(
                        adv_flat, i.flat.valid_var, name="adv_valid")
                    # Optionally normalize advantages
                    eps = tf.constant(1e-8, dtype=tf.float32)
                    if self.center_adv:
                        with tf.name_scope("center_adv"):
                            mean, var = tf.nn.moments(adv_valid, axes=[0])
                            adv_valid = tf.nn.batch_normalization(
                                adv_valid, mean, var, 0, 1, eps)
                    if self.positive_adv:
                        with tf.name_scope("positive_adv"):
                            m = tf.reduce_min(adv_valid)
                            adv_valid = (adv_valid - m) + eps
                    # Not supporting rnn at this point..(TODO)
                    policy_dist_info_flat = dist_infos_flat[idx]
                    policy_dist_info_valid = filter_valids_dict(
                        policy_dist_info_flat,
                        i.flat.valid_var,
                        name="policy_dist_info_valid")

                with tf.name_scope("surr_loss"):
                    lr = pol_dist.likelihood_ratio_sym(
                        i.valid.action_var,
                        i.valid.policy_old_dist_info_vars,
                        policy_dist_info_valid,
                        name="lr")

                    surr_vanilla = lr * adv_valid
                    losses.append(surr_vanilla)

        surr_loss_mean = tf.reduce_mean(losses)

        with tf.name_scope("kl"):
            policy_dist_info_flat = self.policy.wrapped_policy.dist_info_sym(
                kl_inputs.flat.obs_var,
                kl_inputs.flat.policy_state_info_vars,
                name="policy_dist_info_flat")
            policy_dist_info_valid = filter_valids_dict(
                policy_dist_info_flat,
                kl_inputs.flat.valid_var,
                name="policy_dist_info_valid")
            kl = pol_dist.kl_sym(
                kl_inputs.valid.policy_old_dist_info_vars,
                policy_dist_info_valid,
            )
            pol_mean_kl = tf.reduce_mean(kl)

        # Only use the vanilla surrogate loss now
        # Not sure what is the proper way to clip this loss
        surr_obj = tf.identity(surr_loss_mean, name="surr_obj")

        return surr_obj, pol_mean_kl

    def _policy_opt_input_values(self, samples_data):
        """ Map rollout samples to the policy optimizer inputs """
        policy_opt_input_values = []
        for i in range(self.policy.n_tasks):
            policy_state_info_list = [
                samples_data[i]["agent_infos"][k] for k in self.policy.state_info_keys
            ]
            policy_old_dist_info_list = [
                samples_data[i]["agent_infos"][k]
                for k in self.policy.distribution.dist_info_keys
            ]

            policy_opt_input_values.append(self._policy_opt_inputs[i]._replace(
                obs_var=samples_data[i]["observations"],
                action_var=samples_data[i]["actions"],
                reward_var=samples_data[i]["rewards"],
                baseline_var=samples_data[i]["baselines"],
                valid_var=samples_data[i]["valids"],
                policy_state_info_vars_list=policy_state_info_list,
                policy_old_dist_info_vars_list=policy_old_dist_info_list,
            ))

        # Input values for computing kl divergence
        # TODO rewrite the reshape part..
        obs_values = np.concatenate([d["observations"] for d in samples_data])
        valid_values = np.concatenate([d["valids"] for d in samples_data])
        policy_state_info_list = []
        for k in self.policy.state_info_keys:
            lst = []
            for i in range(self.policy.n_tasks):
                lst.append(samples_data[i]["agent_infos"][k])
            lst_arr = np.array(lst)
            policy_state_info_list.append(
                np.reshape(lst_arr, newshape=(-1,)+lst_arr.shape[2:4]))
        
        policy_old_dist_info_list = []
        for k in self.policy.distribution.dist_info_keys:
            lst = []
            for i in range(self.policy.n_tasks):
                lst.append(samples_data[i]["agent_infos"][k])
            lst_arr = np.array(lst)
            policy_old_dist_info_list.append(
                np.reshape(lst_arr, newshape=(-1,)+lst_arr.shape[2:4]))

        kl_inputs = self._kl_opt_inputs._replace(
            obs_var=obs_values,
            valid_var=valid_values,
            policy_old_dist_info_vars_list=policy_old_dist_info_list,
            policy_state_info_vars_list=policy_state_info_list,
        )
        return flatten_inputs([policy_opt_input_values, kl_inputs])

    def optimize_policy(self, itr, samples_data):
        policy_opt_input_values = self._policy_opt_input_values(samples_data)

        # Train policy network
        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(policy_opt_input_values)
        # logger.log("Computing KL before")
        self.optimizer.optimize(policy_opt_input_values)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(policy_opt_input_values)
        logger.record_tabular("{}/LossBefore".format(self.policy.name),
                              loss_before)
        logger.record_tabular("{}/LossAfter".format(self.policy.name),
                              loss_after)
        logger.record_tabular("{}/dLoss".format(self.policy.name),
                              loss_before - loss_after)
        # num_traj = self.batch_size // self.max_path_length
        # actions = samples_data["actions"][:num_traj, ...]
        # logger.record_histogram("{}/Actions".format(self.policy.name), actions)

        self._fit_baseline(samples_data)

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        self.start_worker(sess)
        start_time = time.time()
        last_average_return = None
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)
                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                # if self.plot:
                #     self.plotter.update_plot(self.policy, self.max_path_length)
                #     if self.pause_for_plot:
                #         input("Plotting evaluation run: Press Enter to "
                #               "continue...")

        self.shutdown_worker()
        if created_session:
            sess.close()
        return last_average_return

    def _fit_baseline(self, samples_data):
        """ Update baselines from samples. """

        policy_opt_input_values = self._policy_opt_input_values(samples_data)
        for i in range(self.policy.n_tasks):
            # Augment reward from baselines
            rewards_tensor = self.f_rewards[i](*policy_opt_input_values)
            returns_tensor = self.f_returns[i](*policy_opt_input_values)
            returns_tensor = np.squeeze(returns_tensor)

            paths = samples_data[i]["paths"]
            valids = samples_data[i]["valids"]
            baselines = [path["baselines"] for path in paths]

            # Recompute parts of samples_data
            aug_rewards = []
            aug_returns = []
            for rew, ret, val, path in zip(rewards_tensor, returns_tensor, valids,
                                           paths):
                path["rewards"] = rew[val.astype(np.bool)]
                path["returns"] = ret[val.astype(np.bool)]
                aug_rewards.append(path["rewards"])
                aug_returns.append(path["returns"])
            aug_rewards = tensor_utils.concat_tensor_list(aug_rewards)
            aug_returns = tensor_utils.concat_tensor_list(aug_returns)
            samples_data[i]["rewards"] = aug_rewards
            samples_data[i]["returns"] = aug_returns

            # Calculate explained variance
            ev = special.explained_variance_1d(
                np.concatenate(baselines), aug_returns)
            logger.record_tabular("Baseline/ExplainedVariance", ev)

            # Fit baseline
            logger.log("Fitting baseline...")
            if hasattr(self.baseline, "fit_with_samples"):
                self.baseline.fit_with_samples(paths, samples_data[i])
            else:
                self.baseline.fit(paths)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

    def _policy_adapt_opt_values(self, samples_data):
        policy_state_info_list = [
            samples_data["agent_infos"][k] for k in self.policy.state_info_keys
        ]
        policy_old_dist_info_list = [
            samples_data["agent_infos"][k]
            for k in self.policy.distribution.dist_info_keys
        ]
        values = self._policy_opt_inputs[0]._replace(
            obs_var=samples_data["observations"],
            action_var=samples_data["actions"],
            reward_var=samples_data["rewards"],
            baseline_var=samples_data["baselines"],
            valid_var=samples_data["valids"],
            policy_state_info_vars_list=policy_state_info_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_list,
        )
        return flatten_inputs(values)

    def adapt_policy(self, n_itr=1, sess=None):

        assert n_itr > 0

        if sess is None:
            sess = tf.get_default_session()
            sess = tf.Session() if not sess else sess

        self.start_worker(sess)
        policy_params = self.policy.get_params_internal()
        values_before_adapt = sess.run(policy_params)

        for itr in range(n_itr):
            with logger.prefix('itr #%d | ' % itr):
                logger.log('Obtaining samples...')
                paths = self.obtain_samples(itr)
                logger.log('Processing samples...')
                samples_data = self.process_samples(itr, paths)
                values = self._policy_adapt_opt_values(samples_data)
                logger.log('Computing adapted policy parameters...')
                params = self.f_adapt(*values)
                self.policy.update_params(params)

        # Revert the policy as not adapted
        self.policy.update_params(values_before_adapt)

        return params
