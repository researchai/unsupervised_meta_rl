import abc

from garage.contrib.exp.core import Snapshotable


class Agent(abc.ABC, Snapshotable):
    def __init__(self):
        pass
