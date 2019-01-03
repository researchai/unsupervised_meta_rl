import numpy as np
import torch

from garage.contrib.exp.core import Policy
from garage.contrib.exp.agents import Agent


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
        if baseline:
            raise NotImplementedError

        self.env_spec = env_spec
        self.policy = policy
        self.discount = discount

        self.policy_pi_opt = torch.optim.Adam(self.policy.parameters())

    def get_actions(self, obs):
        self.policy.eval()
        actions, _ = self.policy.sample(obs)
        return actions

    def train_once(self, samples):
        samples = self._process_sample(samples)

        pi_loss = -(samples['logp_pi'], samples['adv']).mean()
        self.policy.train()
        self.policy_pi_opt.zero_grad()
        pi_loss.backward()
        self.policy_pi_opt.step()

    def _process_sample(self, samples):
        self.policy.eval()
        logp_pi_all = np.array([], dtype=np.float32)
        adv_all = np.array([], dtype=np.float32)

        for path in samples:
            obs = path['observations']
            actions = path['actions']
            rews = path['rewards']

            logp_pi = self.policy.logpdf(obs, actions)
            advs = np.zeros_like(rews)

            discounted_adv = 0
            for i in reversed(range(len(rews))):
                discounted_adv = rews[i] + self.discount * discounted_adv
                advs[i] = discounted_adv

            logp_pi_all += logp_pi
            adv_all += advs

        return {
            'logp_pi': logp_pi_all,
            'adv': adv_all
        }
