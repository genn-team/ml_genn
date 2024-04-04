from __future__ import annotations

from typing import Tuple, Union
from .serialisers import Serialiser

from .utils.module import get_object
from .utils.value import set_values

from .serialisers import default_serialisers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Connection, Population

SerialiserInitializer = Union[Serialiser, str]


class Network:
    """Unstructured network model
    
    Attributes:
        populations:    List of all populations in network
        connections:    List of all connections in network
    
    Args:
        default_params: Default parameters to use for neuron and synapse
                        models created within the scope of this network.
                        These are typically provided by the compiler.
    """
    _context = None

    def __init__(self, default_params: dict = {}):
        self.default_params = default_params
        self.populations = []
        self.connections = []

    def load(self, keys: Tuple = (),
             serialiser: SerialiserInitializer = "numpy"):
        """Load network state from checkpoints

        Args:
            keys:       tuple of keys used to select correct checkpoint.
                        Typically might contain epoch number or configuration.
            serialiser: Serialiser to load checkpoints with (should be the 
                        same type of serialiser which was used to create them)
        """
        # Create serialiser
        serialiser = get_object(serialiser, Serialiser, "Serialiser",
                                default_serialisers)

        # Loop through connections
        for c in self.connections:
            # Deserialize everthing relating to connection
            state = serialiser.deserialise_all(keys + (c,))
            
            # Set any variables in copnnectivity
            # **TODO** synapse
            # **TODO** give error/warning if variable not found
            set_values(c.connectivity, state)
        
        # Loop through populations
        for p in self.populations:
            # Deserialize everthing relating to population
            state = serialiser.deserialise_all(keys + (p,))

            # Set any variables in neuron
            # **TODO** give error if variable was not found
            set_values(p.neuron, state)

    @staticmethod
    def _add_population(pop: Population):
        if Network._context is None:
            raise RuntimeError("Population must be created "
                               "inside a ``with network:`` block")
        Network._context.populations.append(pop)

    @staticmethod
    def _add_connection(conn: Connection):
        if Network._context is None:
            raise RuntimeError("Connection must be created "
                               "inside a ``with network:`` block")
        Network._context.connections.append(conn)

    @staticmethod
    def get_default_params(type):
        if Network._context is None:
            return {}
        else:
            return Network._context.default_params.get(type, {})

    def __enter__(self):
        if Network._context is not None:
            raise RuntimeError("Nested networks are not currently supported")

        Network._context = self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert Network._context is not None
        Network._context = None
