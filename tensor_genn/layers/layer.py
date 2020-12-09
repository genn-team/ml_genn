from tensor_genn.layers.base_layer import BaseLayer
from tensor_genn.layers.neurons import Neurons
from tensor_genn.layers.if_neurons import IFNeurons

class Layer(BaseLayer):

    def __init__(self, name, neurons=None):
        super(Layer, self).__init__(name, neurons)
        if self.neurons is None:
            self.neurons = IFNeurons()
        if not isinstance(neurons, Neurons):
            raise ValueError('"Layer" class instances require "Neuron" class neuron groups')

    def connect(self, sources, synapses):
        if len(sources) != len(synapses):
            raise ValueError('sources list and synapse list length mismatch')

        for source, synapse in zip(sources, synapses):
            synapse.connect(source, self)

    def set_weights(self, weights):
        if len(weights) != len(self.upstream_synapses):
            raise ValueError('weight matrix list and upsteam synapse list length mismatch')

        for synapse, w in zip(self.upstream_synapses, weights):
            synapse.set_weights(w)

    def get_weights(self):
        return [synapse.get_weights() for synapse in self.upstream_synapses]
