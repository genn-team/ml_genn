from tensor_genn.layers.base_layer import BaseLayer
from tensor_genn.layers.input_neurons import InputNeurons
from tensor_genn.layers.poisson_input_neurons import PoissonInputNeurons

class InputLayer(BaseLayer):

    def __init__(self, name, shape, neurons=None):
        super(InputLayer, self).__init__(name, neurons)
        if self.neurons is None:
            self.neurons = PoissonInputNeurons()
        if not isinstance(neurons, InputNeurons):
            raise ValueError('"InputLayer" class instances require "InputNeuron" class neuron groups')
        self.neurons.shape = shape

    def set_input_batch(self, data_batch):
        self.neurons.set_input_batch(data_batch)
