# pylint: disable=attribute-defined-outside-init, no-self-use, too-many-statementss
"""PEARL implementation in Pytorch.

Code is adapted from https://github.com/katerakelly/oyster.
"""

from collections import OrderedDict
import copy

from dowel import logger, tabular
import numpy as np
import torch

from garage import log_performance, log_multitask_performance, TrajectoryBatch
from garage.misc.tensor_utils import discount_cumsum
from garage.np.algos.meta_rl_algorithm import MetaRLAlgorithm
from garage.replay_buffer.meta_replay_buffer import MetaReplayBuffer
from garage.sampler.pearl_sampler import PEARLSampler
import garage.torch.utils as tu


class PEARLSAC(MetaRLAlgorithm):
    """A PEARL model based on https://arxiv.org/abs/1903.08254.

    PEARL, which stands for Probablistic Embeddings for Actor-Critic
    Reinforcement Learning, is an off-policy meta-RL algorithm. It is built
    on top of SAC using two Q-functions and a value function with an addition
    of an inference network that estimates the posterior p(z|c). The policy
    is conditioned on the latent variable Z in order to adpat its behavior to
    specific tasks.

    Args:
        env (object): Meta-RL Environment.
        nets (list): A list containing policy, Q-function, and value function
            networks.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int): Number of tasks for testing.
        latent_dim (int): Size of latent context vector.
        policy_lr (float): Policy learning rate.
        qf_lr (float): Q-function learning rate.
        vf_lr (float): Value function learning rate.
        context_lr (float): Inference network learning rate.
        policy_mean_reg_coeff (float): Policy mean regulation weight.
        policy_std_reg_coeff (float): Policy std regulation weight.
        policy_pre_activation_coeff (float): Policy pre-activation weight.
        soft_target_tau (float): Interpolation parameter for doing the
            soft target update.
        kl_lambda (float): KL lambda value.
        optimizer_class (callable): Type of optimizer for training networks.
        recurrent (bool): Whether or not context encoder is recurrent.
        use_information_bottleneck (bool): False means latent context is
            deterministic.
        use_next_obs_in_context (bool): Whether or not to use next observation
            in distinguishing between tasks.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_steps_posterior (int): Number of transitions to obtain per task
            with z ~ posterior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        num_evals (int): Number of independent evaluations to perform.
        num_steps_per_eval (int): Number of transitions to evaluate on.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        max_path_length (int): Maximum path length.
        discount (float): RL discount factor.
        replay_buffer_size (int): Maximum samples in replay buffer.
        reward_scale (int): Reward scale.
        num_exp_traj_eval (int): Number of trajectories collected before
            posterior sampling at test time.
        update_post_train (int): How often to resample context when obtaining
            data during training (in trajectories).
        eval_deterministic (bool): Whether to make policy deterministic during
            evaluation.
        render_eval_paths (bool): Whether or not to render paths during
            evaluation.
        plotter (object): Plotter.

    """

    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            vf,
            num_train_tasks,
            num_test_tasks,
            latent_dim,
            test_env=None,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3E-4,
            policy_mean_reg_coeff=1E-3,
            policy_std_reg_coeff=1E-3,
            policy_pre_activation_coeff=0.,
            soft_target_tau=0.005,
            kl_lambda=.1,
            optimizer_class=torch.optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            meta_batch_size=64,
            num_steps_per_epoch=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=0,
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
            num_exp_traj_eval=2,
            update_post_train=1,
            eval_deterministic=True,
            render_eval_paths=False,
            multi_env=False,
    ):

        self.env = env
        self.policy = policy
        self._qf1 = qf1
        self._qf2 = qf2
        self._vf = vf
        self._num_train_tasks = num_train_tasks
        self._num_test_tasks = num_test_tasks
        self._latent_dim = latent_dim

        self._policy_mean_reg_coeff = policy_mean_reg_coeff
        self._policy_std_reg_coeff = policy_std_reg_coeff
        self._policy_pre_activation_coeff = policy_pre_activation_coeff
        self._soft_target_tau = soft_target_tau
        self._kl_lambda = kl_lambda
        self._recurrent = recurrent
        self._use_information_bottleneck = use_information_bottleneck
        self._use_next_obs_in_context = use_next_obs_in_context

        self._meta_batch_size = meta_batch_size
        self._num_steps_per_epoch = num_steps_per_epoch
        self._num_initial_steps = num_initial_steps
        self._num_tasks_sample = num_tasks_sample
        self._num_steps_prior = num_steps_prior
        self._num_steps_posterior = num_steps_posterior
        self._num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self._num_evals = num_evals
        self._num_steps_per_eval = num_steps_per_eval
        self._batch_size = batch_size
        self._embedding_batch_size = embedding_batch_size
        self._embedding_mini_batch_size = embedding_mini_batch_size
        self._max_path_length = max_path_length
        self._discount = discount
        self._replay_buffer_size = replay_buffer_size
        self._reward_scale = reward_scale
        self._num_exp_traj_eval = num_exp_traj_eval
        self._update_post_train = update_post_train

        self._eval_deterministic = eval_deterministic
        self._render_eval_paths = render_eval_paths

        self._total_env_steps = 0
        self._total_train_steps = 0
        self._eval_statistics = None
        self._multi_env = multi_env

        self.sampler = PEARLSampler(
            env=env,
            policy=policy,
            max_path_length=max_path_length,
        )

        self._num_total_tasks = num_train_tasks + num_test_tasks
        if test_env is None:
            self.test_env = env
            tasks = env.sample_tasks(self._num_total_tasks)
            self._train_tasks = tasks[:num_train_tasks]
            self._test_tasks = tasks[num_train_tasks:]
        else:
            self.test_env = test_env
            self._train_tasks = env.sample_tasks(num_train_tasks)
            self._test_tasks = self.test_env.sample_tasks(num_test_tasks)
        
        self._train_tasks_idx = range(num_train_tasks)
        self._test_tasks_idx = range(num_test_tasks)

        # buffer for training RL update
        self.replay_buffer = {
            i: MetaReplayBuffer(max_replay_buffer_size=replay_buffer_size,
                                observation_dim=env.observation_space.low.size,
                                action_dim=env.action_space.low.size)
            for i in self._train_tasks_idx
        }
        # buffer for training encoder update
        self.context_replay_buffer = {
            i: MetaReplayBuffer(max_replay_buffer_size=replay_buffer_size,
                                observation_dim=env.observation_space.low.size,
                                action_dim=env.action_space.low.size)
            for i in self._train_tasks_idx
        }

        self.target_vf = copy.deepcopy(self._vf)
        self.vf_criterion = torch.nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.networks[1].parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self._qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self._qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self._vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.policy.networks[0].parameters(),
            lr=context_lr,
        )

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        data = self.__dict__.copy()
        del data['replay_buffer']
        del data['context_replay_buffer']
        return data

    def train(self, runner):
        """Obtain samples, train, and evaluate for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
        for _ in runner.step_epochs():
            epoch = runner.step_itr / self._num_steps_per_epoch

            # obtain initial set of samples from all train tasks
            if epoch == 0:
                for idx in self._train_tasks_idx:
                    self.task_idx = idx
                    self._task = self._train_tasks[idx]
                    self.env.set_task(self._task)
                    self.env.reset()
                    self.obtain_samples(self._num_initial_steps, 1, np.inf)

            # obtain samples from random tasks
            for _ in range(self._num_tasks_sample):
                idx = np.random.randint(len(self._train_tasks_idx))
                self.task_idx = idx
                self._task = self._train_tasks[idx]
                self.env.set_task(self._task)
                self.env.reset()
                self.context_replay_buffer[idx].clear()

                # obtain samples with z ~ prior
                if self._num_steps_prior > 0:
                    self.obtain_samples(self._num_steps_prior, 1, np.inf)
                # obtain samples with z ~ posterior
                if self._num_steps_posterior > 0:
                    self.obtain_samples(self._num_steps_posterior, 1,
                                        self._update_post_train)
                # obtain extras samples for RL training but not encoder
                if self._num_extra_rl_steps_posterior > 0:
                    self.obtain_samples(self._num_extra_rl_steps_posterior,
                                        1,
                                        self._update_post_train,
                                        add_to_enc_buffer=False)

            logger.log('Training...')
            # sample train tasks and optimize networks
            for _ in range(self._num_steps_per_epoch):
                indices = np.random.choice(self._train_tasks_idx,
                                           self._meta_batch_size)
                self.train_once(indices)
                self._total_train_steps += 1
                runner.step_itr += 1

            logger.log('Evaluating...')
            # evaluate
            self.evaluate(epoch)

    def train_once(self, indices):
        """Perform one step of training.

        Args:
            indices (list): Tasks used for training.

        """
        mb_size = self._embedding_mini_batch_size
        num_updates = self._embedding_batch_size // mb_size

        # sample context
        context_batch = self.sample_context(indices)
        # clear context and hidden encoder state
        self.policy.reset_belief(num_tasks=len(indices))

        # only loop for recurrent encoder to truncate backprop
        for i in range(num_updates):
            context = context_batch[:, i * mb_size:i * mb_size + mb_size, :]
            self.optimize_policy(indices, context)
            self.policy.detach_z()

    def optimize_policy(self, indices, context):
        """Perform algorithm optimizing.

        Args:
            indices (list): Tasks used for training.
            context (torch.tensor): Context data.

        """
        num_tasks = len(indices)

        # data shape is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)
        policy_outputs, task_z = self.policy(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flatten out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize qf and encoder networks
        q1_pred = self._qf1(torch.cat([obs, actions], dim=1), task_z)
        q2_pred = self._qf2(torch.cat([obs, actions], dim=1), task_z)
        v_pred = self._vf(obs, task_z.detach())

        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self._use_information_bottleneck:
            kl_div = self.policy.compute_kl_div()
            kl_loss = self._kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        rewards_flat = rewards.view(self._batch_size * num_tasks, -1)
        rewards_flat = rewards_flat * self._reward_scale
        terms_flat = terms.view(self._batch_size * num_tasks, -1)
        q_target = rewards_flat + (
            1. - terms_flat) * self._discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target)**2) + torch.mean(
            (q2_pred - q_target)**2)
        #qf_loss.backward(retain_graph=True)
        qf_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        q1 = self._qf1(torch.cat([obs, new_actions], dim=1), task_z.detach())
        q2 = self._qf2(torch.cat([obs, new_actions], dim=1), task_z.detach())
        min_q = torch.min(q1, q2)

        # optimize vf
        v_target = min_q - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()
        
        # optimize policy
        log_policy_target = min_q
        policy_loss = (log_pi - log_policy_target).mean()

        mean_reg_loss = self._policy_mean_reg_coeff * (policy_mean**2).mean()
        std_reg_loss = self._policy_std_reg_coeff * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self._policy_pre_activation_coeff * (
            (pre_tanh_value**2).sum(dim=1).mean())
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # log stats
        if self._eval_statistics is None:
            self._eval_statistics = OrderedDict()
            if self._use_information_bottleneck:
                z_mean = np.mean(np.abs(tu.to_numpy(self.policy.z_means[0])))
                z_sig = np.mean(tu.to_numpy(self.policy.z_vars[0]))
                self._eval_statistics['TrainZMean'] = z_mean
                self._eval_statistics['TrainZVariance'] = z_sig
                self._eval_statistics['KLDivergence'] = tu.to_numpy(kl_div)
                self._eval_statistics['KLLoss'] = tu.to_numpy(kl_loss)
            self._eval_statistics['QFLoss'] = np.mean(tu.to_numpy(qf_loss))
            self._eval_statistics['VFLoss'] = np.mean(tu.to_numpy(vf_loss))
            self._eval_statistics['PolicyLoss'] = np.mean(
                tu.to_numpy(policy_loss))

    def evaluate(self, epoch):
        """Evaluate train and test tasks.

        Args:
            epoch (int): Current epoch.

        """
        if self._eval_statistics is None:
            self._eval_statistics = OrderedDict()

        # evaluate on a subset of train tasks
        indices = np.random.choice(self._train_tasks_idx,
                                   len(self._test_tasks_idx))
        # evaluate train tasks with posterior sampled from the training replay buffer
        """
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self._task = self._train_tasks[idx]
            self.env.set_task(self._task)
            self.env.reset()
            paths = []
            for _ in range(self._num_steps_per_eval // self._max_path_length):
                context = self.sample_context(idx)
                self.policy.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(
                    deterministic=self._eval_deterministic,
                    max_samples=self._max_path_length,
                    accum_context=False,
                    max_trajs=1,
                    resample=np.inf)
                paths += p

            returns = np.mean([sum(path['rewards']) for path in paths])
            train_returns.append(returns)
        train_returns = np.mean(train_returns)
        # eval train tasks with on-policy data to match eval of test tasks
        avg_train_return, train_success_rate = self.get_average_returns(
            indices, False)
        """
        self.get_average_returns(indices, False, epoch)
        # eval test tasks
        self.get_average_returns(self._test_tasks_idx, True, epoch)

        # log stats
        self.policy.log_diagnostics(self._eval_statistics)
        #self._eval_statistics['ZTrainAverageReturn'] = train_returns
        #self._eval_statistics['TrainAverageReturn'] = avg_train_return
        #self._eval_statistics['AverageReturn'] = avg_test_return
        #if train_success_rate is not None:
        #    self._eval_statistics['TrainSuccessRate'] = train_success_rate
        #if test_success_rate is not None:
        #    self._eval_statistics['SuccessRate'] = test_success_rate

        # record values
        for key, value in self._eval_statistics.items():
            tabular.record(key, value)
        self._eval_statistics = None

        tabular.record('Iteration', epoch)
        tabular.record('TotalTrainSteps', self._total_train_steps)
        tabular.record('TotalEnvSteps', self._total_env_steps)

    def get_average_returns(self, indices, test, epoch):
        """Get average returns for specific tasks.

        Args:
            indices (list): List of tasks.

        Returns:
            float: Average return.
            float: Success rate.

        """
        discounted_returns = []
        undiscounted_returns = []
        completion = []
        success = []
        traj = []
        for idx in indices:
            
            eval_paths = []
            for _ in range(self._num_evals):
                paths = self.collect_paths(idx, test)
                paths[-1]['terminals'] = paths[-1]['terminals'].squeeze()
                paths[-1]['dones'] = paths[-1]['terminals']
                #paths[-1]['env_infos']['task'] = paths[-1]['env_infos']['task']['velocity']
                eval_paths.append(paths[-1])
                discounted_returns.append(discount_cumsum(paths[-1]['rewards'], self._discount))
                undiscounted_returns.append(sum(paths[-1]['rewards']))
                completion.append(float(paths[-1]['terminals'].any()))
                # calculate success rate for metaworld tasks
                if 'success' in paths[-1]['env_infos']:
                    success.append(paths[-1]['env_infos']['success'].any())
            
            temp_traj = TrajectoryBatch.from_trajectory_list(self.env, eval_paths)
            traj.append(temp_traj)
        
        if test:
            with tabular.prefix("Test/"):
                if self._multi_env:
                    log_multitask_performance(epoch, TrajectoryBatch.concatenate(*traj), 
                        self._discount, task_names=self.test_env.task_names)
                log_performance(epoch, TrajectoryBatch.concatenate(*traj), 
                    self._discount, prefix='Average')
        else:
            with tabular.prefix("Train/"):
                if self._multi_env:
                    log_multitask_performance(epoch, TrajectoryBatch.concatenate(*traj), 
                        self._discount, task_names=self.env.task_names)
                log_performance(epoch, TrajectoryBatch.concatenate(*traj), 
                    self._discount, prefix='Average')
        
        

        """
        avg_discounted_return = np.mean([rtn[0] for rtn in discounted_returns])
        tabular.record('NumTrajs', len(discounted_returns))
        tabular.record('AverageDiscountedReturn', avg_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('CompletionRate', np.mean(completion))
        if success:
            tabular.record('SuccessRate', np.mean(success))
        """

        

    def obtain_samples(self,
                       num_samples,
                       resample_z_rate,
                       update_posterior_rate,
                       add_to_enc_buffer=True):
        """Obtain samples.

        Args:
            num_samples (int): Number of samples to obtain.
            update_posterior_rate (int): How often to update q(z|c) from which
                z is sampled (in trajectories).
            resample_z_rate (int): How often (in trajectories) to resample
                context.
            add_to_enc_buffer (bool): Whether or not to add samples to encoder buffer.

        """
        self.policy.reset_belief()
        num_transitions = 0

        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(
                max_samples=num_samples - num_transitions,
                max_trajs=update_posterior_rate,
                accum_context=False,
                resample=resample_z_rate)
            num_transitions += n_samples
            for path in paths:
                self.replay_buffer[self.task_idx].add_path(path)

                if add_to_enc_buffer:
                    self.context_replay_buffer[self.task_idx].add_path(path)

            
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.policy.infer_posterior(context)

        self._total_env_steps += num_transitions

    def sample_data(self, indices):
        """Sample batch of training data from a list of tasks.

        Args:
            indices (list): List of tasks.

        Returns:
            torch.Tensor: Data.

        """
        # transitions sampled randomly from replay buffer
        batches = [
            tu.np_to_pytorch_batch(
                self.replay_buffer[idx].sample_batch(self._batch_size))
            for idx in indices
        ]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        """Sample batch of context from a list of tasks.

        Args:
            indices (list): List of tasks.

        Returns:
            torch.Tensor: Context data.

        """
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [
            tu.np_to_pytorch_batch(
                self.context_replay_buffer[idx].sample_batch(self._batch_size)) 
            for idx in indices
        ]
        context = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        if self._use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    def unpack_batch(self, batch):
        """Unpack a batch and return individual elements.

        Args:
            batch (torch.Tensor): Data.

        Returns:
            torch.Tensor: Unpacked data.

        """
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def collect_paths(self, idx, test):
        """Collect paths for evaluation.

        Args:
            idx (int): Task to collect paths from.

        Returns:
            list: A list containing paths.

        """
        self.task_idx = idx
        if test: 
            self._task = self._test_tasks[idx]
            self.test_env.set_task(self._task)
            self.sampler.env = self.test_env
            self.test_env.reset()
        else:
            self._task = self._train_tasks[idx]
            self.env.set_task(self._task)
            self.env.reset()
        self.policy.reset_belief()
        paths = []
        num_transitions = 0
        num_trajs = 0

        while num_transitions < self._num_steps_per_eval:
            path, num = self.sampler.obtain_samples(
                deterministic=self._eval_deterministic,
                max_samples=self._num_steps_per_eval - num_transitions,
                max_trajs=1,
                accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self._num_exp_traj_eval:
                context = self.policy.context
                self.policy.infer_posterior(context)

        self.sampler.env = self.env
        return paths

    def _update_target_network(self):
        """Update parameters in the target vf network."""
        for target_param, param in zip(self.target_vf.parameters(),
                                       self._vf.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._soft_target_tau) \
                    + param.data * self._soft_target_tau
            )

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return self.policy.networks + [self.policy] + [
            self._qf1, self._qf2, self._vf, self.target_vf
        ]

    def get_exploration_policy(self):
        return copy.deepcopy(self.policy)

    def adapt_policy(self, exploration_policy, exploration_trajectories):
        pass

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.
        """
        if device is None:
            device = tu.device
        for net in self.networks:
            net.to(device)
