from enum import Enum
import numpy as np
from tensor_genn.genn_models import spike_input_model
from tensor_genn.genn_models import poisson_input_model
from tensor_genn.genn_models import if_input_model


class InputType(Enum):
    SPIKE = 'spike'
    POISSON = 'poisson'
    IF = 'if'


class Input(object):

    def __init__(self, model, params, vars_init, global_params,
                 name, shape):
        self.name = name
        self.shape = shape
        self.model = model
        self.params = params
        self.vars_init = vars_init
        self.global_params = global_params

        self.downstream_layers = []

        self.tg_model = None


    def compile(self, tg_model):
        self.tg_model = tg_model

        nrn_n = np.prod(self.shape)
        for batch_i in range(tg_model.batch_size):

            # Add neuron population
            nrn_name = '{}_nrn_{}'.format(self.name, batch_i)
            nrn = tg_model.g_model.add_neuron_population(
                nrn_name, nrn_n, self.model, self.params, self.vars_init
            )
            for gp in self.global_params:
                nrn.set_extra_global_param(gp, self.global_params[gp])


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
            nrn_name = '{}_nrn_{}'.format(self.name, batch_i)
            nrn = self.tg_model.g_model.neuron_populations[nrn_name]
            if batch_i < n_samples:
                nrn.vars['input'].view[:] = data_batch[batch_i].flatten()
            else:
                nrn.vars['input'].view[:] = np.zeros(input_size)
            nrn.push_state_to_device()


class SpikeInput(Input):
    def __init__(self, name, shape):
        super(SpikeInput, self).__init__(
            spike_input_model, {}, {'input': 0.0}, {},
            name, shape
        )


class PoissonInput(Input):
    def __init__(self, name, shape):
        super(PoissonInput, self).__init__(
            poisson_input_model, {}, {'input': 0.0}, {},
            name, shape
        )


class IFInput(Input):
    def __init__(self, name, shape):
        super(IFInput, self).__init__(
            if_input_model, {}, {'input': 0.0, 'Vmem': 0.0}, {},
            name, shape
        )
