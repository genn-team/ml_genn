from ml_genn.layers.base_layer import BaseLayer
from ml_genn.layers.input_neurons import InputNeurons
from ml_genn.layers.poisson_input_neurons import PoissonInputNeurons

class InputLayer(BaseLayer):

    def __init__(self, name, shape, neurons=PoissonInputNeurons()):
        super(InputLayer, self).__init__(name, neurons)
        self.neurons.shape = shape

        if not isinstance(self.neurons, InputNeurons):
            raise ValueError('"InputLayer" class instances require "InputNeuron" class neurons')

    def set_input_batch(self, data_batch):
        self.neurons.set_input_batch(data_batch)
