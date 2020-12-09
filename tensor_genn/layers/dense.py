from tensor_genn.layers import Layer, DenseSynapses

class Dense(Layer):

    def __init__(self, name, units, neurons=None):
        super(Dense, self).__init__(name, neurons)
        self.units = units

    def connect(self, sources):
        synapses = [DenseSynapses(self.units) for i in range(len(sources))]
        super(Dense, self).connect(sources, synapses)
