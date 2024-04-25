import numpy as np

from abc import ABC
from numbers import Number
from typing import List, Tuple
from ..utils.model import NeuronModel

from abc import abstractmethod


class Readout(ABC):
    """Base class for all readouts"""
    
    @abstractmethod
    def add_readout_logic(self, model: NeuronModel, **kwargs) -> NeuronModel:
        """Create a copy of a neuron model with any additional state 
        and functionality required to implement this readout added.

        Args:
            model:  Base neuron model
        """
        pass

    @abstractmethod
    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        """Read out the value from the state of a compiled neuron group.
        
        Args:
            genn_pop:   GeNN ``NeuronGroup`` object population
                        with readout has been compiled into
            batch_size: Batch size of model readout is part of
            shape:      Shape of population
        """
        pass

    @property
    def reset_vars(self) -> List[Tuple[str, str, Number]]:
        """Get list of tuples describing name, type and value
        to reset any state variables added by readout to
        """
        return []
