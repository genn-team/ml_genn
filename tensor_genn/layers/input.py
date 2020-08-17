from enum import Enum
import numpy as np

from tensor_genn.layers.base_layer import BaseLayer
from tensor_genn.layers.neuron_models import spike_input_model
from tensor_genn.layers.neuron_models import poisson_input_model
from tensor_genn.layers.neuron_models import if_input_model


class InputType(Enum):
    SPIKE = 'spike'
    POISSON = 'poisson'
    IF = 'if'


class Input(BaseLayer):

    def __init__(self, model, params, vars_init, global_params, name, shape):
        super(Input, self).__init__(model, params, vars_init, global_params, name)
        self.shape = shape


    def compile(self, tg_model):
        super(Input, self).compile(tg_model)


    def set_input_batch(self, data_batch):
        # Input sanity check
        n_samples = data_batch.shape[0]
        if n_samples > self.tg_model.batch_size:
            raise ValueError('sample count {} > batch size {}'.format(n_samples, self.tg_model.batch_size))
        sample_size = np.prod(data_batch.shape[1:])
        input_size = np.prod(self.shape)
        if sample_size != input_size:
            raise ValueError('sample size {} != input size {}'.format(sample_size, input_size))

        # Set inputs
        for batch_i in range(self.tg_model.batch_size):
            if batch_i < n_samples:
                self.nrn[batch_i].vars['input'].view[:] = data_batch[batch_i].flatten()
            else:
                self.nrn[batch_i].vars['input'].view[:] = np.zeros(input_size)
            self.nrn[batch_i].push_state_to_device()


class SpikeInput(Input):
    def __init__(self, name, shape):
        super(SpikeInput, self).__init__(
            spike_input_model, {}, {'input': 0.0}, {}, name, shape
        )


class PoissonInput(Input):
    def __init__(self, name, shape):
        super(PoissonInput, self).__init__(
            poisson_input_model, {}, {'input': 0.0}, {}, name, shape
        )


class IFInput(Input):
    def __init__(self, name, shape):
        super(IFInput, self).__init__(
            if_input_model, {}, {'input': 0.0, 'Vmem': 0.0}, {}, name, shape
        )
