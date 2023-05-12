from abc import ABC, abstractmethod


class Controller(ABC):
    def __init__(self, node_id, parent, init: bool = False):
        self.node_id = node_id
        self.parent = parent

    @abstractmethod
    def update(self, delta_time: float):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def mutate(self, mutation_rate: float, mutation_sigma: float):
        pass
