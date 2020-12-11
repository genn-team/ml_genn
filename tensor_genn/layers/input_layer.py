from tensor_genn.layers.base_layer import BaseLayer
from tensor_genn.layers.input_neurons import InputNeurons
from tensor_genn.layers.poisson_input_neurons import PoissonInputNeurons

class InputLayer(BaseLayer):

    def __init__(self, name, shape, neurons=PoissonInputNeurons()):
        super(InputLayer, self).__init__(name, neurons)
        assert(isinstance(self.neurons, InputNeurons))
        self.neurons.shape = shape

    def set_input_batch(self, data_batch):
        self.neurons.set_input_batch(data_batch)
