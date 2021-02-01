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
        input_view = self.neurons.nrn.vars['input'].view

        # Add batch dimension if missing
        if len(input_view.shape) == 1:
            input_view = input_view.reshape(1, -1)

        # Check batch dimension
        if data_batch.shape[0] != input_view.shape[0]:
            raise ValueError('data batch {} != input batch {}'.format(data_batch.shape[0], input_view.shape[0]))

        # Check input dimensions
        if data_batch.shape[1:] != self.shape:
            raise ValueError('data shape {} != input shape {}'.format(data_batch.shape[1:], self.shape))

        input_view[:] = data_batch.reshape(input_view.shape[0], -1)
        self.neurons.nrn.push_state_to_device()
