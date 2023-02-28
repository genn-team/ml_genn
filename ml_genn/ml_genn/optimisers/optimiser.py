from abc import ABC

from abc import abstractmethod


class Optimiser(ABC):
    @abstractmethod
    def set_step(self, genn_cu, step: int):
        pass

    @abstractmethod
    def get_model(self, gradient_ref, var_ref, zero_gradient: bool):
        pass
