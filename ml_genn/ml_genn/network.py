from .serialisers import Serialiser

from .utils.module import get_object
from .utils.value import set_values

from .serialisers import default_serialisers

class Network:
    _context = None

    def __init__(self):
        self.populations = []
        self.connections = []

    @staticmethod
    def add_population(pop):
        if Network._context is None:
            raise RuntimeError("Population must be created "
                               "inside a ``with network:`` block")
        Network._context.populations.append(pop)

    @staticmethod
    def add_connection(conn):
        if Network._context is None:
            raise RuntimeError("Connection must be created "
                               "inside a ``with network:`` block")
        Network._context.connections.append(conn)

    def load(self, keys=(), serialiser="numpy"):
        # Create serialiser
        serialiser = get_object(serialiser, Serialiser, "Serialiser",
                                default_serialisers)

        # Loop through connections
        for c in self.connections:
            # Deserialize everthing relating to connection
            state = serialiser.deserialise_all(keys + (c,))
            
            # **HACK** for now, assume single serialised variable is weight
            if len(state) > 0:
                assert len(state) == 1
                c.connectivity.weight = next(iter(state.values()))
        
        # Loop through populations
        for p in self.populations:
            # Deserialize everthing relating to population
            state = serialiser.deserialise_all(keys + (p,))

            # Set any variables in neuron
            # **TODO** also synapse, also give error if variable was not found
            set_values(p.neuron, state)

    def __enter__(self):
        if Network._context is not None:
            raise RuntimeError("Nested networks are not currently supported")

        Network._context = self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert Network._context is not None
        Network._context = None
