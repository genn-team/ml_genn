from ml_genn.layers.base_layer import BaseLayer
from ml_genn.layers.input_neurons import InputNeurons
from ml_genn.layers.poisson_input_neurons import PoissonInputNeurons

class InputLayer(BaseLayer):

    def __init__(self, name, shape, neurons=PoissonInputNeurons()):
        if not isinstance(neurons, InputNeurons):
            raise ValueError('"InputLayer" class instances require "InputNeuron" class neurons')

        super(InputLayer, self).__init__(name, neurons)
        self.shape = shape

    def set_input_batch(self, data_batch):
        if data_batch.shape[1:] != self.shape:
            raise ValueError('data shape {} != layer shape {}'.format(data_batch.shape[1:], self.shape))

        self.neurons.set_input_batch(data_batch)
