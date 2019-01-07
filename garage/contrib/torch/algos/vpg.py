import numpy as np
import torch

from garage.contrib.exp.agents import Agent
from garage.contrib.exp.core import Policy
from garage.misc.special import discount_cumsum


class VPG(Agent):
    """
    Vanilla Policy Gradient.

    TODO: Implement baseline support.
          Testing.
    """

    def __init__(self,
                 env_spec,
                 policy: Policy,
                 discount,
                 baseline,
                 *args,
                 **kwargs):

        self.env_spec = env_spec
        self.policy = policy
        self.discount = discount
        self.baseline = baseline

        self.policy_pi_opt = torch.optim.Adam(self.policy.parameters())

    def get_actions(self, obs):
        self.policy.eval()
        actions, _ = self.policy.sample(obs)
        return actions

    def train_once(self, paths):
        logp_pi, adv = self._process_sample(paths)

        pi_loss = -(logp_pi * adv).mean()
        self.policy.train()
        self.policy_pi_opt.zero_grad()
        pi_loss.backward()
        self.policy_pi_opt.step()

    def _process_sample(self, paths):
        self.policy.eval()
        logp_pi_all = torch.empty((0, ))
        adv_all = np.array([], dtype=np.float32)

        # Add 'return' to paths required by baseline
        for path in paths:
            rews = path['rewards']
            path['returns'] = discount_cumsum(rews, self.discount)
        self.baseline.fit(paths)

        for path in paths:
            obs = torch.Tensor(path['observations'])
            actions = torch.Tensor(path['actions']).view(-1, 1)
            logp_pi = self.policy._logpdf(obs, actions)

            rtns = path['returns']
            baselines = self.baseline.predict(path)
            advs = rtns - baselines

            logp_pi_all = torch.cat((logp_pi_all, logp_pi))
            adv_all = np.concatenate((adv_all, advs))

        return logp_pi_all, torch.Tensor(adv_all)

    def get_summary(self):
        pass
