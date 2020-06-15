import torch
import torch.nn.functional as F
import numpy as np
from dowel import tabular

from garage.torch.algos import SAC
import garage.torch.utils as tu


class DIAYN(SAC):
    def __init__(self,
                 env_spec,
                 skills_num,
                 discriminator,
                 policy,
                 qf1,
                 qf2,
                 replay_buffer,
                 *,  # Everything after this is numbers
                 max_path_length,
                 max_eval_path_length=None,
                 gradient_steps_per_itr,
                 fixed_alpha=None,  # empirically could be 0.1
                 target_entropy=None,
                 initial_log_entropy=0.,
                 discount=0.99,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 target_update_tau=5e-3,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 reward_scale=1.0,
                 optimizer=torch.optim.Adam,
                 steps_per_epoch=1,
                 num_evaluation_trajectories=10,
                 eval_env=None):

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf1=qf1,
                         qf2=qf2,
                         replay_buffer=replay_buffer,
                         max_path_length=max_path_length,
                         max_eval_path_length=max_eval_path_length,
                         gradient_steps_per_itr=gradient_steps_per_itr,
                         fixed_alpha=fixed_alpha,
                         target_entropy=target_entropy,
                         initial_log_entropy=initial_log_entropy,
                         discount=discount,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         target_update_tau=target_update_tau,
                         policy_lr=policy_lr,
                         qf_lr=qf_lr,
                         reward_scale=reward_scale,
                         optimizer=optimizer,
                         steps_per_epoch=steps_per_epoch,
                         num_evaluation_trajectories=num_evaluation_trajectories,
                         eval_env=eval_env)

        self._skills_num = skills_num
        self._prob_skill = np.full(skills_num, 1.0 / skills_num)
        self._discriminator = discriminator
        self._discriminator_optimizer = self._optimizer(
            self._discriminator.parameters(),
            lr=self._policy_lr)

    def train(self, runner):
        if not self._eval_env:
            self._eval_env = runner.get_env_copy()
        last_return = None

        for _ in runner.step_epoches():
            for _ in range(self.steps_per_epoch):  # step refers to episode ?

                if not self._buffer_prefilled:
                    batch_size = int(self.min_buffer_size)
                else:
                    batch_size = None
                runner.step_path = runner.obtain_samples(
                    runner.step_itr, batch_size)
                path_returns = []

                for path in runner.step_path:
                    reward = self._obtain_pseudo_reward(path['state'],
                                                        path['skill']).reshape(
                        -1, 1),
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             state=path['states'],
                             next_state=path['next_states'],
                             skill=path['skills'],
                             reward=reward,
                             next_observation=path['next_observations'],
                             terminal=path['dones'].reshape(-1, 1)))
                    path_returns.append(sum(reward))
                assert len(path_returns) is len(runner.step_path)
                self.episode_rewards.append(np.mean(path_returns))
                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss, discriminator_loss = \
                        self._learn_once()
                # last_return = self._evaluate_policy(runner.step_itr)
                self._log_statistics(policy_loss, qf1_loss, qf2_loss, discriminator_loss)
                tabular.record('TotalEnvSteps', runner.total_env_steps)
                runner.step_itr += 1

    def _learn_once(self, itr=None, paths=None):
        del itr
        del paths
        if self._buffer_prefilled:
            samples = self.replay_buffer.sample_transitions(
                self.buffer_batch_size)
            samples = tu.dict_np_to_torch(samples)
            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(0, samples)
            self._update_targets()
            discriminator_loss = self.optimize_discriminator(samples)

        return policy_loss, qf1_loss, qf2_loss, discriminator_loss

    def optimize_policy(self, itr, samples_data):
        states = samples_data['state']
        skills = samples_data['skill']
        qf1_loss, qf2_loss = self._critic_objective(samples_data)

        self._qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self._qf1_optimizer.step()

        self._qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self._qf2_optimizer.step()

        action_dists = self._policy(states, skills)
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions)
        self._policy_optimizer.zero_grad()
        policy_loss.backward()

        self._policy_optimizer.step()

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        return policy_loss, qf1_loss, qf2_loss

    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):
        states = samples_data['state']
        skills = samples_data['skill']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        min_q_new_actions = torch.min(self._qf1(states, new_actions, skills),
                                      self._qf2(states, new_actions, skills))
        policy_objective = ((alpha * log_pi_new_actions) -
                            min_q_new_actions.flatten()).mean()
        return policy_objective

    def _critic_objective(self, samples_data):
        states = samples_data['state']
        actions = samples_data['action']
        rewards = samples_data['reward'].flatten()
        terminals = samples_data['terminal'].flatten()
        next_states = samples_data['next_state']
        skills = samples_data['skill']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        q1_pred = self._qf1(states, actions, skills)
        q2_pred = self._qf2(states, actions, skills)

        new_next_actions_dist = self._policy(next_states, skills)
        new_next_actions_pre_tanh, new_next_actions = (
            new_next_actions_dist.rsample_with_pre_tanh_value())
        new_log_pi = new_next_actions_dist.log_prob(
            value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)

        target_q_values = torch.min(
            self._target_qf1(next_states, new_next_actions, skills),
            self._target_qf2(
                next_states, new_next_actions, skills)).flatten() - (alpha * new_log_pi)
        with torch.no_grad():
            q_target = rewards * self.reward_scale + (
                1. - terminals) * self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        return qf1_loss, qf2_loss

    # def _get_log_alpha(self, samples_data):
    #     pass

    # def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):

    # def _critic_objective(self, samples_data):

    def _discriminator_objective(self, samples_data):
        state = samples_data['next_state']
        skill = samples_data['skill']

        discriminator_pred = self._discriminator(state)
        discriminator_target = self._get_one_hot_tensor(self._skills_num,
                                                        skill)

        discriminator_loss = F.mse_loss(discriminator_pred.flatten(),
                                        discriminator_target)

        return discriminator_loss

    # def _update_targets(self):

    def optimize_discriminator(self, samples_data):
        discriminator_loss = self._discriminator_objective(samples_data)

        self._discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self._discriminator_optimizer.step()

        return discriminator_loss

    # def optimize_policy(self, itr, samples_data):

    def _evaluate_policy(self, epoch):
        pass
        # TODO: test how it improves the training of sac

        # def _temperature_objective(self, log_pi, samples_data):
        # pass

    def _log_statistics(self, policy_loss, qf1_loss, qf2_loss, discriminator_loss):
        with torch.no_grad():
            tabular.record('AlphaTemperature/mean',
                           self._log_alpha.exp().mean().item())
        tabular.record('Discriminator/Loss', float(discriminator_loss))
        tabular.record('Policy/Loss', policy_loss.item())
        tabular.record('QF/{}'.format('Qf1Loss'), float(qf1_loss))
        tabular.record('QF/{}'.format('Qf2Loss'), float(qf2_loss))
        tabular.record('ReplayBuffer/buffer_size',
                       self.replay_buffer.n_transitions_stored)
        tabular.record('Average/TrainAverageReturn',
                       np.mean(self.episode_rewards))

    @property
    def networks(self):
        return [
            self._policy, self._discriminator, self._qf1, self._qf2,
            self._target_qf1, self._target_qf2
        ]

    # def to(self, device=None):

    def _sample_skill(self):  # to maximize entropy
        return np.random.choice(self._skills_num, self._prob_skill)

    def _obtain_pseudo_reward(self, state, skill):
        q_z = self._discriminator(state)[-1, skill]
        reward = np.log(q_z) - np.log(np.full(q_z.shape, self._prob_skill[
            skill]))  # TODO: is it working?
        return reward

    def _get_one_hot_tensor(self, size, indices):  # size-length vector  # TODO: need to change N-D one_hot
        return np.eye(size)[indices]
