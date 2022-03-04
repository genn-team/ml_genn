from weakref import proxy

from .neurons import Neuron

class Connection:
    def __init__(self, source, target):
        # Store weak references to source and target in class
        self.source = weakref(source)
        self.target = weakref(target)

        # Add weak references to ourselves to source
        # and target's outgoing and incoming connection lists
        source.outgoing_connections.append(weakref(self))
        target.incoming_connections.append(weakref(self))

        # Add connection to model
        Model.add_connection(self)
