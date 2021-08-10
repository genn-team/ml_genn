from ml_genn.layers import ConnectivityType
from ml_genn.layers import Layer, IdentitySynapses
from ml_genn.layers.if_neurons import IFNeurons

class Identity(Layer):

    def __init__(self, name, connectivity_type='procedural', neurons=IFNeurons()):
        super(Identity, self).__init__(name, neurons)
        self.connectivity_type = ConnectivityType(connectivity_type)

    def connect(self, sources):
        synapses = [IdentitySynapses() for i in range(len(sources))]
        super(Identity, self).connect(sources, synapses)
