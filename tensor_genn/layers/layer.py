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
        self.nrn = None


    def connect(self, sources, connections):
        if len(sources) != len(connections):
            raise ValueError('sources list and connections list length mismatch')

        for source, connection in zip(sources, connections):
            connection.connect(source, self)


    def set_weights(self, weights):
        if len(weights) != len(self.upstream_connections):
            raise ValueError('weight matrix list and upsteam connection list length mismatch')

        for connection, w in zip(self.upstream_connections, weights):
            connection.set_weights(w)

    def get_weights(self):
        return [connection.get_weights() for connection in self.upstream_connections]


    def compile(self, tg_model):
        self.tg_model = tg_model
        self.nrn = [None] * tg_model.batch_size

        nrn_n = np.prod(self.shape)
        for batch_i in range(tg_model.batch_size):

            # Add neuron population
            nrn_name = '{}_nrn_{}'.format(self.name, batch_i)
            self.nrn[batch_i] = tg_model.g_model.add_neuron_population(
                nrn_name, nrn_n, self.model, self.params, self.vars_init
            )
            for gp in self.global_params:
                self.nrn[batch_i].set_extra_global_param(gp, self.global_params[gp])

        for connection in self.upstream_connections:
            connection.compile(tg_model)


class IFLayer(Layer):

    def __init__(self, name, threshold=1.0):
        super(IFLayer, self).__init__(
            name, if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold}
        )

        self.threshold = threshold


    def set_threshold(self, threshold):
        if not self.tg_model:
            raise RuntimeError('model must be compiled before calling set_threshold')

        for batch_i in range(self.tg_model.batch_size):
            nrn_name = '{}_nrn_{}'.format(self.name, batch_i)
            nrn = self.tg_model.g_model.neuron_populations[nrn_name]
            nrn.extra_global_params['Vthr'].view[:] = threshold

        self.threshold = threshold
