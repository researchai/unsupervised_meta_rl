from garage.contrib.exp import Agent
from garage.contrib.exp import Logger
from garage.contrib.exp import Observer
from garage.contrib.exp import Snapshotor


class Experiment():
    def __init__(
            self,
            observer: Observer,
            agent: Agent,
        snapshotor: Snapshotor,
            logger: Logger,
            # common experiment variant,
            n_itr,
            batch_size,
            max_path_length,
            discount,
    ):
        pass

    def train(self):
        pass
