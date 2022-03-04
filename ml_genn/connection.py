from . import synapses

from weakref import proxy
from .connectivity import Connectivity
from .model import Model
from .synapses import Synapse

from .utils.model import get_module_models, get_model

# Use Keras-style trick to get dictionary containing default synapse models
_synapse_models = get_module_models(synapses, Synapse)

class Connection:
    def __init__(self, source, target, connectivity: Connectivity, 
                 synapse="delta", add_to_model=True):
        # Store weak references to source and target in class
        self.source = proxy(source)
        self.target = proxy(target)

        self.connectivity = connectivity
        self.synapse = get_model(synapse, Synapse, "Synapse", _synapse_models)
        
        # Add weak references to ourselves to source
        # and target's outgoing and incoming connection lists
        source.outgoing_connections.append(proxy(self))
        target.incoming_connections.append(proxy(self))

        # Run connectivity-specific connection logic
        # e.g. automatically-calculating population sizes
        self.connectivity.connect(source, target)

        # Add connection to model
        if add_to_model:
            Model.add_connection(self)
