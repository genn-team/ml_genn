from itertools import count
from typing import Optional, Union
from weakref import ref
from .connectivity import Connectivity
from .network import Network
from .population import Population
from .synapses import Synapse

from copy import deepcopy
from .utils.module import get_object

from .synapses import default_synapses

SynapseInitializer = Union[Synapse, str]
ConnectivityInitializer = Union[Connectivity, str]


class Connection:
    _new_id = count()

    def __init__(self, source: Population, target: Population,
                 connectivity: ConnectivityInitializer,
                 synapse: SynapseInitializer = "delta", 
                 name: Optional[str] = None, add_to_model: bool = True):
        # Store weak references to source and target in class
        self.source = ref(source)
        self.target = ref(target)

        self.connectivity = deepcopy(connectivity)
        self.synapse = get_object(synapse, Synapse, "Synapse",
                                  default_synapses)

        # Generate unique name if required
        self.name = (f"Conn{next(Connection._new_id)}" if name is None
                     else name)

        # Add weak references to ourselves to source
        # and target's outgoing and incoming connection lists
        source.outgoing_connections.append(ref(self))
        target.incoming_connections.append(ref(self))

        # Run connectivity-specific connection logic
        # e.g. automatically-calculating population sizes
        self.connectivity.connect(source, target)

        # Add connection to model
        if add_to_model:
            Network.add_connection(self)
