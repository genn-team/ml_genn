from itertools import count
from typing import Optional, List, Sequence, Union
from .network import Network
from .neurons import Neuron

from .utils.module import get_object

from .neurons import default_neurons
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Connection


def _get_shape(shape):
    if shape is None or isinstance(shape, Sequence):
        return shape
    elif isinstance(shape, int):
        return (shape,)
    else:
        raise RuntimeError("Population shapes should either be left "
                           "unspecified with None or specified as an "
                           "integer or a sequence")


Shape = Union[None, int, Sequence[int]]
NeuronInitializer = Union[Neuron, str]


class Population:
    """A population of neurons

    Attributes:
        shape:          Shape of population
        name:           Name of connection (only really used 
                        for debugging purposes)
        record_spikes:  Should spikes from this population be recorded?
                        This is required to subsequently attach SpikeRecorder
                        callbacks to this population
    """
    _new_id = count()

    def __init__(self, neuron: NeuronInitializer, shape: Shape = None,
                 record_spikes: bool = False, 
                 record_spike_events: bool = False,
                 name: Optional[str] = None, add_to_model: bool = True):
        self.neuron = neuron
        self.shape = _get_shape(shape)
        self._incoming_connections = []
        self._outgoing_connections = []
        self.record_spikes = record_spikes
        self.record_spike_events = record_spike_events

        # Generate unique name if required
        self.name = (f"Pop{next(Population._new_id)}" if name is None
                     else name)

        # Add population to model
        if add_to_model:
            Network._add_population(self)

    # **TODO** shape setter which validate shape with neuron parameters etc

    @property
    def incoming_connections(self) -> List["Connection"]:
        """Incoming connections to this population"""
        return self._incoming_connections

    @property
    def outgoing_connections(self) -> List["Connection"]:
        """Outgoing connections fromt his population"""
        return self._outgoing_connections

    @property
    def neuron(self) -> Neuron:
        """The neuron model to use for this population

        Can be specified as either a Neuron object or, 
        for built in neuron models whose constructors 
        require no arguments, a string e.g. "leaky_integrate_fire"
        """
        return self._neuron

    @neuron.setter
    def neuron(self, n: NeuronInitializer):
        self._neuron = get_object(n, Neuron, "Neuron", default_neurons)

    def __str__(self):
        return self.name
