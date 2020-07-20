import numpy as np

class Layer(object):

    def __init__(self, name, model, params, vars_init, global_params):
        self.name = name
        self.model = model
        self.params = params
        self.vars_init = vars_init
        self.global_params = global_params

        self.downstream_connections = []
        self.upstream_connections = []
        self.shape = None
        self.tg_model = None


    def set_weights(self, weights):
        if len(weights) != len(self.upstream_connections):
            raise ValueError('weight matrix list and upsteam connection list length mismatch')

        for connection, w in zip(self.upstream_connections, weights):
            connection.set_weights(w)

    def get_weights(self):
        return [connection.get_weights() for connection in self.upstream_connections]


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

        for connection in self.upstream_connections:
            connection.compile(tg_model)
