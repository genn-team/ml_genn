import numpy as np

from abc import ABC
from ..utils.model import NeuronModel

from abc import abstractmethod


class Readout(ABC):
    @abstractmethod
    def add_readout_logic(self, model: NeuronModel) -> NeuronModel:
        pass

    @abstractmethod
    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        pass

    @property
    def reset_vars(self):
        return []
