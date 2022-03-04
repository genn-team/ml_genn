from typing import Union

from . import Model
from .neurons import Neuron

class Population:
    def __init__(self, neuron: Neuron, size: Union[None, int]=None):
        self.neuron = neuron
        self.size = size

        # Add population to model
        Model.add_population(self)
