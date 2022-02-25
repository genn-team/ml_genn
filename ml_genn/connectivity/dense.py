from ml_genn.layers import Layer, DenseSynapses
from ml_genn.layers.if_neurons import IFNeurons

class Dense(Layer):

    def __init__(self, name, units, neurons=IFNeurons()):
        super(Dense, self).__init__(name, neurons)
        self.units = units

    def connect(self, sources):
        synapses = [DenseSynapses(self.units) for i in range(len(sources))]
        super(Dense, self).connect(sources, synapses)
