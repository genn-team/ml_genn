from abc import ABC
from ..utils.model import NeuronModel

from abc import abstractmethod


class Loss(ABC):
    @abstractmethod
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        pass

    @abstractmethod
    def set_target(self, genn_pop, y_true, shape, batch_size: int, 
                   example_timesteps: int):
        pass
