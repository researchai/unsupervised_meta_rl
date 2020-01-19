# exisitng PEARL uses old version of rlkit SAC (2 q-functions and value function)
# new SAC uses 2 q-functions, and value function is replaced by getting action from
# policy then calculating q value like in DDPG

from collections import OrderedDict
import copy

from dowel import logger, tabular
import numpy as np
import torch

from garage.misc import eval_util
from garage.replay_buffer.multi_task_replay_buffer import MultiTaskReplayBuffer
from garage.sampler.in_place_sampler import PEARLSampler
import garage.torch.utils as tu


class PEARLSAC:
    def __init__(self,
                 env,
                 nets,
                 num_train_tasks,
                 num_eval_tasks,
                 latent_dim,

                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 vf_lr=1e-3,
                 context_lr=1e-3,
                 policy_mean_reg_weight=1e-3,
                 policy_std_reg_weight=1e-3,
                 policy_pre_activation_weight=0.,
                 soft_target_tau=1e-2,
                 kl_lambda=1.,
                 optimizer_class=torch.optim.Adam,
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
        self.num_train_tasks = num_train_tasks
        self.num_eval_tasks = num_eval_tasks
        self.latent_dim = latent_dim

        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
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

        self.sampler = PEARLSampler(
            env=env,
            policy=nets[0],
            max_path_length=self.max_path_length,
        )

        self.num_total_tasks = num_train_tasks + num_eval_tasks
        self.tasks = env.sample_tasks(self.num_total_tasks)
        self.tasks_idx = range(self.num_total_tasks)
        self.train_tasks_idx = list(self.tasks_idx[:num_train_tasks])
        self.eval_tasks_idx = list(self.tasks_idx[-num_eval_tasks:])

        # buffer for training RL update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks_idx,
            )
        # buffer for training encoder update
        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks_idx,
        )

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = copy.deepcopy(self.vf)
        self.vf_criterion = torch.nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.networks[1].parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.policy.networks[0].parameters(),
            lr=context_lr,
        )

    def train(self, runner):
        """Train."""
        # for each epoch, obtain samples from tasks, perform meta-updates, and evaluate
        for _ in runner.step_epochs():
            epoch = runner.step_itr / self.num_steps_per_epoch

            # obtain initial set of samples from all train tasks
            if epoch == 0:
                for idx in self.train_tasks_idx:
                    self.task_idx = idx
                    self.task = self.tasks[idx]
                    self.env.set_task(self.task)
                    self.env.reset()
                    self.obtain_samples(self.num_initial_steps, 1, np.inf)
            
            # obtain samples from random tasks
            for _ in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks_idx))
                self.task_idx = idx
                self.task = self.tasks[idx]
                self.env.set_task(self.task)
                self.env.reset()
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.obtain_samples(self.num_steps_prior, 1, np.inf)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.obtain_samples(self.num_steps_posterior, 1, self.update_post_train)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.obtain_samples(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False)
            
            # sample train tasks and compute gradient updates
            for _ in range(self.num_steps_per_epoch):
                indices = np.random.choice(self.train_tasks_idx, self.meta_batch)
                self.train_once(indices)
                self.total_train_steps += 1
                runner.step_itr += 1

            # evaluate
            self.evaluate(epoch)

    def train_once(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context
        context_batch = self.sample_context(indices)
        # clear context and hidden encoder state
        self.policy.reset_belief(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.optimize_policy(indices, context)
            # stop backprop
            self.policy.detach_z()

    def optimize_policy(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)

        policy_outputs, task_z = self.policy(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flatten out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(torch.cat([obs, actions, task_z], dim=1), None)
        q2_pred = self.qf2(torch.cat([obs, actions, task_z], dim=1), None)
        v_pred = self.vf(obs, task_z.detach())

        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.policy.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # optimize qf and encoder (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)

        qf_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        q1 = self.qf1(torch.cat([obs, new_actions, task_z], dim=1), None)
        q2 = self.qf2(torch.cat([obs, new_actions, task_z], dim=1), None)
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

        # log stats
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(tu.to_numpy(self.policy.z_means[0])))
                z_sig = np.mean(tu.to_numpy(self.policy.z_vars[0]))
                self.eval_statistics['TrainZMean'] = z_mean
                self.eval_statistics['TrainZVariance'] = z_sig
                self.eval_statistics['KLDivergence'] = tu.to_numpy(kl_div)
                self.eval_statistics['KLLoss'] = tu.to_numpy(kl_loss)
            self.eval_statistics['QFLoss'] = np.mean(tu.to_numpy(qf_loss))
            self.eval_statistics['VFLoss'] = np.mean(tu.to_numpy(vf_loss))
            self.eval_statistics['PolicyLoss'] = np.mean(tu.to_numpy(policy_loss))

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        # evaluate on a subset of train tasks
        indices = np.random.choice(self.train_tasks_idx, len(self.eval_tasks_idx))
        # evaluate train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.task = self.tasks[idx]
            self.env.set_task(self.task)
            self.env.reset()
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.sample_context(idx)
                self.policy.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, 
                                                   max_samples=self.max_path_length,
                                                   accum_context=False,
                                                   max_trajs=1,
                                                   resample=np.inf)
                paths += p

            returns = np.mean([sum(path["rewards"]) for path in paths])
            train_returns.append(returns)
        train_returns = np.mean(train_returns)
        # eval train tasks with on-policy data to match eval of test tasks
        avg_train_return, train_success_rate = self.get_average_returns(indices)
        # eval test tasks
        avg_test_return, test_success_rate = self.get_average_returns(self.eval_tasks_idx)

        # save the final posterior
        self.policy.log_diagnostics(self.eval_statistics)
        self.eval_statistics['ZTrainAverageReturn'] = train_returns
        self.eval_statistics['TrainAverageReturn'] = avg_train_return
        self.eval_statistics['TestAverageReturn'] = avg_test_return
        if train_success_rate is not None:
            self.eval_statistics['TrainSuccessRate'] = train_success_rate
        if test_success_rate is not None:
            self.eval_statistics['TestSuccessRate'] = test_success_rate

        # record values
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

    def get_average_returns(self, indices):
        final_returns = []
        final_success = []
        num_paths = 0
        for idx in indices:
            returns = []
            for _ in range(self.num_evals):
                paths = self.collect_paths(idx)
                num_paths += len(paths)
                print("LWD: ", paths[0])
                temp_returns = [np.mean([sum(path["rewards"])]) for path in paths]
                returns.append(temp_returns)

                if 'success' in paths[0]['env_infos']:
                    success = sum([path['env_infos']['success'][-1]
                        for path in paths])
                    final_success.append(success)

            final_returns.append(np.mean([a[-1] for a in returns]))
        if final_success:
            success_rate = sum(final_success) * 100.0 / num_paths
        else:
            success_rate = None
        return np.mean(final_returns), success_rate

    def obtain_samples(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
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

    def sample_data(self, indices):
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

    def collect_paths(self, idx):
        self.task_idx = idx
        self.task = self.tasks[idx]
        self.env.set_task(self.task)
        self.env.reset()
        self.policy.reset_belief()
        paths = []
        num_transitions = 0
        num_trajs = 0

        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                    max_samples=self.num_steps_per_eval - num_transitions,
                                                    max_trajs=1,
                                                    accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                context = self.policy.context
                self.policy.infer_posterior(context)

        return paths

    def _update_target_network(self):
        for target_param, param in zip(self.target_vf.parameters(), self.vf.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) \
                    + param.data * self.soft_target_tau
            )

    @property
    def networks(self):
        return self.policy.networks + [self.policy] + [self.qf1, self.qf2, self.vf, self.target_vf]

    def to(self, device=None):
        if device == None:
            device = tu.device
        for net in self.networks:
            net.to(device)

