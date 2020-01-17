# exisitng PEARL uses old version of rlkit SAC (2 q-functions and value function)
# new SAC uses 2 q-functions, and value function is replaced by getting action from
# policy then calculating q value like in DDPG

import akro
from collections import OrderedDict
import copy

from dowel import logger, tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage.envs import EnvSpec
from garage.misc import eval_util
from garage.replay_buffer.multi_task_replay_buffer import MultiTaskReplayBuffer
from garage.sampler.in_place_sampler import InPlaceSampler
from garage.torch.policies import TanhGaussianMLPPolicy2
from garage.torch.q_functions import ContinuousMLPQFunction
import garage.torch.utils as tu


class PEARLSAC:
    def __init__(self,
                 env,
                 nets,
                 train_tasks,
                 eval_tasks,
                 latent_dim,

                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 context_lr=1e-3,
                 policy_mean_reg_weight=1e-3,
                 policy_std_reg_weight=1e-3,
                 policy_pre_activation_weight=0.,
                 alpha=None,
                 target_entropy=None,
                 initial_log_entropy=0.,
                 use_automatic_entropy_tuning=False,
                 soft_target_tau=5e-3,
                 kl_lambda=1.,
                 optimizer=torch.optim.Adam,
                 recurrent=False,
                 use_information_bottleneck=True,
                 use_next_obs_in_context=False,

                 meta_batch=64,
                 num_steps_per_epoch=1000,
                 num_initial_steps=100,
                 num_tasks_sample=100,
                 num_steps_prior=100,
                 num_steps_posterior=100,
                 num_extra_rl_steps_posterior=100,
                 num_evals=10,
                 num_steps_per_eval=1000,
                 batch_size=1024,
                 embedding_batch_size=1024,
                 embedding_mini_batch_size=1024,
                 max_path_length=1000,
                 discount=0.99,
                 replay_buffer_size=1000000,
                 reward_scale=1,
                 num_exp_traj_eval=1,
                 update_post_train=1,

                 eval_deterministic=True,
                 render_eval_paths=False,
                 render=False,
                 plotter=None,
                 ):

        # meta params
        self.env = env
        self.policy = nets[0]
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.latent_dim = latent_dim

        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.initial_log_entropy = initial_log_entropy
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.soft_target_tau = soft_target_tau
        self.kl_lambda = kl_lambda
        self.recurrent = recurrent
        self.use_information_bottleneck = use_information_bottleneck
        self.use_next_obs_in_context = use_next_obs_in_context

        self.meta_batch = meta_batch
        self.num_steps_per_epoch = num_steps_per_epoch
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.num_exp_traj_eval = num_exp_traj_eval
        self.update_post_train = update_post_train
        
        self.eval_deterministic = eval_deterministic
        self.render_eval_paths = render_eval_paths
        self.render = render
        self.plotter = plotter

        self.total_env_steps = 0
        self.total_train_steps = 0
        self.eval_statistics = None

        self.sampler = InPlaceSampler(
            env=env,
            policy=nets[0],
            max_path_length=self.max_path_length,
        )

        # buffer for training RL update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )
        # buffer for training encoder update
        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
        )

        self.qf1, self.qf2 = nets[1:3]
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.policy_optimizer = optimizer(
            self.policy.networks[1].parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.context_optimizer = optimizer(
            self.policy.networks[0].parameters(),
            lr=context_lr,
        )
        if self.use_automatic_entropy_tuning and not alpha:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = torch.tensor([self.initial_log_entropy], dtype=torch.float, requires_grad=True)
            self.alpha_optimizer = optimizer([self.log_alpha], lr=policy_lr)
        else:
            self.alpha = alpha


    def train(self, runner):
        """Train."""
        # for each epoch, collect data from tasks, perform meta-updates, and evaluate
        for _ in runner.step_epochs():
            epoch = runner.step_itr / self.num_steps_per_epoch

            # collect initial set of data for train and eval
            if epoch == 0:
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf)
            
            # sample data from train tasks
            for _ in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False)
            
            # sample train tasks and compute gradient updates on parameters
            for _ in range(self.num_steps_per_epoch):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices)
                self.total_train_steps += 1
                runner.step_itr += 1

            # eval
            self.evaluate(epoch)

    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)
        # zero out context and hidden encoder state
        self.policy.reset_belief(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._optimize(indices, context)
            # stop backprop
            self.policy.detach_z()

    def _optimize(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z = self.policy(obs, context)
        next_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        new_policy_outputs, new_task_z = self.policy(next_obs, context)
        new_next_actions, _, _, new_log_pi = new_policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = new_next_actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q networks update
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, new_task_z)
        q2_pred = self.qf2(obs, actions, new_task_z)
        #v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates

        # compute min Q on the new actions
        alpha = self.log_alpha.detach().exp()
        q1 = self.target_qf1(obs, new_next_actions, new_task_z.detach())
        q2 = self.target_qf2(obs, new_next_actions, new_task_z.detach())
        min_q = torch.min(q1, q2)
        target_q_values = min_q - (alpha * new_log_pi)

        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)

        with torch.no_grad():
            q_target = rewards_flat + (1. - terms_flat) * self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.qf2_optimizer.step()

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.policy.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward()
        self.context_optimizer.step()

        # policy update
        alpha = self.log_alpha.detach().exp()
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        log_policy_target = torch.min(q1, q2)
        policy_loss = ((alpha * log_pi) - log_policy_target).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.use_automatic_entropy_tuning:
            alpha_loss = (-(self.log_alpha)* (log_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._update_target_networks()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(tu.to_numpy(self.policy.z_means[0])))
                z_sig = np.mean(tu.to_numpy(self.policy.z_vars[0]))
                self.eval_statistics['TrainZMean'] = z_mean
                self.eval_statistics['TrainZVariance'] = z_sig
                self.eval_statistics['KLDivergence'] = tu.to_numpy(kl_div)
                self.eval_statistics['KLLoss'] = tu.to_numpy(kl_loss)

            self.eval_statistics['QF1Loss'] = np.mean(tu.to_numpy(qf1_loss))
            self.eval_statistics['QF2Loss'] = np.mean(tu.to_numpy(qf2_loss))
            self.eval_statistics['PolicyLoss'] = np.mean(tu.to_numpy(
                policy_loss
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'Q1Predictions',
                tu.to_numpy(q1_pred),
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'Q2Predictions',
                tu.to_numpy(q2_pred),
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'LogPi',
                tu.to_numpy(log_pi),
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'PolicyMu',
                tu.to_numpy(policy_mean),
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'PolicyLogSTD',
                tu.to_numpy(policy_log_std),
            ))

    def evaluate(self, epoch):

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        logger.log('evaluating on {} train tasks'.format(len(indices)))
        # eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.sample_context(idx)
                self.policy.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length,
                                                        accum_context=False,
                                                        max_trajs=1,
                                                        resample=np.inf)
                paths += p

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        # eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices)
        # eval test tasks
        logger.log('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks)

        # save the final posterior
        self.policy.log_diagnostics(self.eval_statistics)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['TrainTaskReturn'] = train_returns
        self.eval_statistics['TrainTaskAverageReturn'] = avg_train_return
        self.eval_statistics['TestTaskAverageReturn'] = avg_test_return

        for key, value in self.eval_statistics.items():
            tabular.record(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

        tabular.record("Epoch", epoch)
        tabular.record("TotalTrainSteps", self.total_train_steps)
        tabular.record("TotalEnvSteps", self.total_env_steps)

    def _do_eval(self, indices):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        self.policy.reset_belief()
        num_transitions = 0

        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples-num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.policy.infer_posterior(context)

        self.total_env_steps += num_transitions

    def collect_paths(self, idx):
        self.task_idx = idx
        self.env.reset_task(idx)
        self.policy.reset_belief()
        paths = []
        num_transitions = 0
        num_trajs = 0

        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                context = self.policy.context
                self.policy.infer_posterior(context)

        goal = self.env.goal
        for path in paths:
            path['goal'] = goal

        return paths

    def sample_sac(self, indices):
        """Sample batch of training data from a list of tasks for training the actor-critic."""
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [tu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        """Sample batch of context from a list of tasks from the replay buffer."""
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [tu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    def unpack_batch(self, batch):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def _min_q(self, obs, actions, task_z):
        q1 = self.target_qf1(obs, actions, task_z.detach())
        q2 = self.target_qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_networks(self):
        """Update parameters in the target q-functions."""
        target_qfs = [self.target_qf1, self.target_qf2]
        qfs = [self.qf1, self.qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(),
                                      qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self.soft_target_tau) +
                                   param.data * self.soft_target_tau)

    @property
    def networks(self):
        return self.policy.networks + [self.policy] + [self.qf1, self.qf2, self.target_qf1, self.target_qf2]

    def to(self, device=None):
        if device == None:
            device = tu.device
        for net in self.networks:
            net.to(device)

