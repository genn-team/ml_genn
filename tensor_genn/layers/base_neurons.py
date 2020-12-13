import numpy as np

class BaseNeurons(object):

    def __init__(self, model, params, vars_init, global_params):
        self.model = model
        self.params = params
        self.vars_init = vars_init
        self.global_params = global_params
        self.name = None
        self.shape = None
        self.signed_spikes = False
        self.nrn = None

    def compile(self, tg_model):
        self.nrn = [None] * tg_model.batch_size

        # Add batch neuron populations
        nrn_n = np.prod(self.shape)
        for batch_i in range(tg_model.batch_size):
            nrn_name = '{}_nrn_{}'.format(self.name, batch_i)

            self.nrn[batch_i] = tg_model.g_model.add_neuron_population(
                nrn_name, nrn_n, self.model, self.params, self.vars_init
            )
            for gp in self.global_params:
                self.nrn[batch_i].set_extra_global_param(gp, self.global_params[gp])
