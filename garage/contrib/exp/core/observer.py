import abc

from garage.contrib.exp.core.snapshotable import Snapshotable


class Observer(abc.ABC, Snapshotable):
    def __init__(self):
        pass
