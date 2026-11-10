"""Example controller plugin: a small timed open-loop sequence."""
from vlab.plugins import ControllerPlugin


class OpenLoopDemo(ControllerPlugin):
    def __init__(self, config):
        self.tick = 0

    def reset(self, observation):
        self.tick = 0

    def act(self, observation):
        self.tick += 1
        if self.tick < 15:
            return [0.0, 0.0, -0.4, 1.0]
        if self.tick < 30:
            return [0.0, 0.0, 0.0, -1.0]
        return [0.0, 0.0, 0.0, 1.0]


def create_controller(config):
    return OpenLoopDemo(config)