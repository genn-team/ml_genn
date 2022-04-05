from typing import Sequence, Union
from weakref import ref
from .connectivity import Connectivity
from .network import Network
from .population import Population
from .synapses import Synapse

from copy import deepcopy
from .utils.module import get_object

from .synapses import default_synapses

class Connection:
    def __init__(self, source: Population, target: Population,
                 connectivity: Connectivity, synapse="delta", add_to_model=True):
        # Store weak references to source and target in class
        self.source = ref(source)
        self.target = ref(target)

        self.connectivity = deepcopy(connectivity)
        self.synapse = get_object(synapse, Synapse, "Synapse", default_synapses)
        
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
