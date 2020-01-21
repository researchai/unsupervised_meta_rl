# pylint: disable=attribute-defined-outside-init
"""A policy used in training meta reinforcement learning algorithms.

It is used in PEARL (Probabilistic Embeddings for Actor-Critic Reinforcement
Learning). The paper on PEARL can be found at https://arxiv.org/abs/1903.08254.
Code is adapted from https://github.com/katerakelly/oyster.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import garage.torch.utils as tu

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


class ContextConditionedPolicy(nn.Module):
    """A policy that outputs actions based on observation and latent context.

    In PEARL, policies are conditioned on current state and a latent context
    (adaptation data) variable Z. This inference network estimates the
    posterior probability of z given past transitions. It uses context
    information stored in the encoder to infer the probabilistic value of z and
    samples from a policy conditioned on z.

    Args:
        latent_dim (int): Latent context variable dimension.
        context_encoder (garage.torch.algo.RecurrentEncoder): Recurrent or
            permutation-invariant context encoder.
        policy (garage.torch.policies.Policy): Policy used to train the
            network.
        use_ib (bool): True if latent context is notdeterministic; false
            otherwise
        use_next_obs (bool): True if next observation is used in context
            for distinguishing tasks; false otherwise.

    """

    def __init__(self, latent_dim, context_encoder, policy, use_ib,
                 use_next_obs):
        super().__init__()
        self._latent_dim = latent_dim
        self._context_encoder = context_encoder
        self._policy = policy

        self._use_ib = use_ib
        self._use_next_obs = use_next_obs

        # initialize buffers for z distribution and z
        # use buffers so latent context can be saved along with model weights
        # z_means and z_vars are the params for the gaussian distribution
        # over latent task belief maintained in the policy; z is a sample from
        # this distribution that the policy is conditioned on
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.reset_belief()

    def reset_belief(self, num_tasks=1):
        """Reset q(z|c) to the prior and sample a new z from the prior.

        Args:
            num_tasks (int): Number of tasks.

        """
        # reset distribution over z to the prior
        mu = tu.zeros(num_tasks, self._latent_dim)
        if self._use_ib:
            var = tu.ones(num_tasks, self._latent_dim)
        else:
            var = tu.zeros(num_tasks, self._latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_from_belief()
        # reset the context collected so far
        self._context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self._context_encoder.reset(num_tasks)

    def sample_from_belief(self):
        """Sample z using distributions from current means and variances."""
        if self._use_ib:
            posteriors = [
                torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(
                    torch.unbind(self.z_means), torch.unbind(self.z_vars))
            ]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def detach_z(self):
        """Disable backprop through z."""
        self.z = self.z.detach()

    def update_context(self, inputs):
        """Append single transition to the current context.

        Args:
            inputs (dict): Dictionary of transition information in np arrays .

        """

        o, a, r, no, d, info = inputs
        o = tu.from_numpy(o[None, None, ...])
        a = tu.from_numpy(a[None, None, ...])
        r = tu.from_numpy(np.array([r])[None, None, ...])
        no = tu.from_numpy(no[None, None, ...])

        if self._use_next_obs:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)

        """
        #LWM
        o, a, r, no, _, _ = inputs
        o = from_numpy(o)
        a = from_numpy(a)
        r = from_numpy(np.array([r]))
        no = from_numpy(no)

        if self._use_next_obs:
            data = torch.cat([o, a, r, no], dim=0)
        else:
            data = torch.cat([o, a, r], dim=0)

        data = torch.unsqueeze(torch.unsqueeze(data, 0), 0)
        """
        if self._context is None:
            self._context = data
        else:
            self._context = torch.cat([self._context, data], dim=1)

    def infer_posterior(self, context):
        """Compute q(z|c) as a function of input context and sample new z.

        Args:
            context (torch.Tensor): Context values.

        """
        params = self._context_encoder(context)
        params = params.view(context.size(0), -1, self._context_encoder._output_dim)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self._use_ib:
            mu = params[..., :self._latent_dim]
            sigma_squared = F.softplus(params[..., self._latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_from_belief()

    # pylint: disable=arguments-differ
    def forward(self, obs, context):
        """Given observations and context, get actions and probs from policy.

        Args:
            obs (torch.Tensor): Observation values.
            context (torch.Tensor): Context values.

        Returns:
            torch.Tensor: Output action values.
            torch.Tensor: z values.

        """
        self.infer_posterior(context)
        self.sample_from_belief()
        task_z = self.z

        # task, batch
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self._policy(in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z

    def get_action(self, obs, deterministic=False):
        """Sample action from the policy, conditioned on the task embedding.

        Args:
            obs (torch.Tensor): Observation values.
            deterministic (bool): Whether or not policy is deterministic.

        Returns:
            torch.Tensor: Output action values.

        """
        z = self.z
        #obs = torch.Tensor(obs)
        #obs = torch.unsqueeze(obs, 0)
        obs = tu.from_numpy(obs[None])
        obs_in = torch.cat([obs, z], dim=1)
        out = self._policy.get_action(obs_in, deterministic=deterministic)
        return out

    def compute_kl_div(self):
        """Compute KL( q(z|c) || p(z) ).

        Returns:
            torch.Tensor: KL( q(z|c) || p(z) ).

        """
        prior = torch.distributions.Normal(tu.zeros(self._latent_dim),
                                           tu.ones(self._latent_dim))
        posteriors = [
            torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(
                torch.unbind(self.z_means), torch.unbind(self.z_vars))
        ]
        kl_divs = [
            torch.distributions.kl.kl_divergence(post, prior)
            for post in posteriors
        ]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def reset(self, dones=None):
        """Reset the environment.

        Args:
            dones (numpy.ndarray): Reset values

        """

    def log_diagnostics(self, eval_statistics):
        """Log data about encodings to eval_statistics dictionary.
       
        Args:
            eval_statistics (dict): Dictionary for logging info.

        """
        z_mean = np.mean(np.abs(tu.to_numpy(self.z_means[0])))
        z_sig = np.mean(tu.to_numpy(self.z_vars[0]))
        eval_statistics['ZMeanEval'] = z_mean
        eval_statistics['ZVarianceEval'] = z_sig

    @property
    def networks(self):
        """Return context_encoder and policy.

        Returns:
            list: Encoder and policy networks.

        """
        return [self._context_encoder, self._policy]
  
    @property
    def context(self):
        """Return context.

        Returns:
            torch.Tensor: context values.

        """
        return self._context
