import abc


class Agent(abc.ABC):
    def __init__(self):
        pass

    def get_actions(self, states):
        pass

    # TODO: `sampels` does not distinguish trajectory.
    # Keep an eye on algorithm that accounts for samples
    # from different trajectories.
    def train_once(self, samples):
        pass
