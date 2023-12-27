from abc import ABC, abstractmethod


class AgentManagerInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def init_episode(self):
        pass

    @abstractmethod
    def run_episode(self, env):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def inference(self, n_steps):
        pass
