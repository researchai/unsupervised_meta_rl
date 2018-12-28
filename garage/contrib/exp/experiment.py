import gym

from garage.contrib.exp import Agent
from garage.contrib.exp import Logger
from garage.contrib.exp import Snapshotor


class Experiment():
    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
            snapshotor: Snapshotor,
            logger: Logger,
            sampler=None,
            # common experiment variant,
            n_itr=10,
            batch_size=1000,
            max_path_length=100,
            discount=0.99,
    ):
        if sampler is None:
            from garage.sampler import BatchSampler
            sampler = BatchSampler(env=env, max_path_length=max_path_length)

        self.env = env
        self.agent = agent
        self.snapshotor = snapshotor
        self.logger = logger
        self.sampler = sampler
        self.n_itr = n_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount

    def train(self):
        itr = 0

        while itr <= self.n_itr:
            itr = itr + 1

            states = self.sampler.reset()
            while self.sampler.sample_count < self.batch_size:
                actions = self.agent.act(states)
                states = self.sampler.step(actions)

            self.agent.train_once(self.sampler.get_samples())
