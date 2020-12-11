from tensor_genn.layers.base_layer import BaseLayer
from tensor_genn.layers.neurons import Neurons
from tensor_genn.layers.if_neurons import IFNeurons

class Layer(BaseLayer):

    def __init__(self, name, neurons=IFNeurons()):
        super(Layer, self).__init__(name, neurons)
        assert(isinstance(self.neurons, Neurons))

    def connect(self, sources, synapses):
        assert(len(sources) == len(synapses))
        for source, synapse in zip(sources, synapses):
            synapse.connect(source, self)

    def set_weights(self, weights):
        assert(len(weights) == len(self.upstream_synapses))
        for synapse, w in zip(self.upstream_synapses, weights):
            synapse.set_weights(w)

    def get_weights(self):
        return [synapse.get_weights() for synapse in self.upstream_synapses]
