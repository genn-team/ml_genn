import numpy as np

class BaseNeurons(object):

    def __init__(self, model, params, vars_init, global_params):
        self.model = model
        self.params = params
        self.vars_init = vars_init
        self.global_params = global_params
        self.signed_spikes = False
        self.nrn = None

    def compile(self, mlg_model, mlg_layer):
        name = '{}_nrn'.format(mlg_layer.name)
        n = np.prod(mlg_layer.shape)

        self.nrn = mlg_model.g_model.add_neuron_population(
            name, n, self.model, self.params, self.vars_init
        )
        for gp in self.global_params:
            self.nrn.set_extra_global_param(gp, self.global_params[gp])
