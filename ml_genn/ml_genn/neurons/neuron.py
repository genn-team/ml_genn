from __future__ import annotations

import numpy as np

from abc import ABC
from typing import Optional, TYPE_CHECKING
from warnings import warn
from ..readouts import Readout
from ..utils.model import NeuronModel

from abc import abstractmethod
from ..utils.module import get_object

from ..readouts import default_readouts

if TYPE_CHECKING:
    from .. import Population

class Neuron(ABC):
    """Base class for all neuron models

    Attributes:
        readout: Type of readout to attach to this neuron's output variable
    """
    def __init__(self, readout: Optional[Readout] = None, **kwargs):
        super(Neuron, self).__init__(**kwargs)
        self.readout = readout

    @abstractmethod
    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        """Gets PyGeNN implementation of neuron model

        Args:
            population: Population this neuron is to be attached to
            dt :        Timestep of simulation (in ms)
            batch_size: Batch size of the model
        """
        pass

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        """Use readout object associated with neuon to
        read output from PyGeNN neuron group into numpy array

        Args:
            genn_pop: PyGeNN neuron group
            batch_size: Batch size of compiled model
            shape: Shape of population
        """
        if self.readout is None:
            raise RuntimeError("Cannot get readout from neuron "
                               "without readout strategy")
        else:
            return self.readout.get_readout(genn_pop, batch_size, shape)

    @property
    def readout(self):
        """Optional object which can be used to
        provide a readout from neuron

        Can be specified as either a Readout object or, 
        for built in readout models whose constructors 
        require no arguments, a string e.g. "spike_count"
        """
        return self._readout

    @readout.setter
    def readout(self, r):
        self._readout = get_object(r, Readout, "Readout",
                                   default_readouts)
