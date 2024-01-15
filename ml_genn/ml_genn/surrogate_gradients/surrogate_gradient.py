from abc import ABC
from ..neurons import Neuron
from ..utils.model import WeightUpdateModel

from abc import abstractmethod

class SurrogateGradient:
    @abstractmethod
    def add_to_weight_update(self, surrogate_var_name: str,
                             weight_update_model: WeightUpdateModel,
                             target_neuron: Neuron):
        pass
