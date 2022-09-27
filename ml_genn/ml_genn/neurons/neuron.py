from abc import ABC
from typing import Optional
from ..readouts import Readout
from ..utils.model import NeuronModel

from abc import abstractmethod
from ..utils.module import get_object

from ..readouts import default_readouts
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Population

class Neuron(ABC):
    """Base class for all neuron models
    
    Attributes:
        softmax: Should softmax of output should be computed
        readout: Object used to provide a readout from neuron
    """
    def __init__(self, softmax: bool = False, 
                 readout: Optional[Readout] = None, **kwargs):
        super(Neuron, self).__init__(**kwargs)
        self.softmax = softmax
        self.readout = get_object(readout, Readout, "Readout",
                                  default_readouts)

    @abstractmethod
    def get_model(self, population: "Population", dt: float) -> NeuronModel:
        """Gets model implementing this neuron
        
        Args:
            population: Population this neuron is to be attached to
            dt : Timestep of simulation (in ms)

        Returns:
            NeuronModel: PyGeNN implementation of neuron
        """
        pass

    def get_readout(self, genn_pop, batch_size: int, shape):
        """Use readout object associated with neuon to
        read output from PyGeNN neuron group
        
        Args:
            genn_pop: PyGeNN neuron group
            batch_size: Batch size of compiled model
            shape: Shape of population
        Returns:
            Numpy array containing read out value
        """
        if self.readout is None:
            raise RuntimeError("Cannot get readout from neuron "
                               "without readout strategy")
        else:
            return self.readout.get_readout(genn_pop, batch_size, shape)
