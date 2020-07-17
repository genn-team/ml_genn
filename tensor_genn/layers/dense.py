import numpy as np
from pygenn.genn_wrapper import NO_DELAY
from tensor_genn.genn_models import if_model


class Dense(object):

    def __init__(self, model, params, vars_init, global_params,
                 name, units):
        self.name = name
        self.units = units
        self.model = model
        self.params = params
        self.vars_init = vars_init
        self.global_params = global_params

        self.downstream_layers = []
        self.upstream_layer = None
        self.weights = None
        self.shape = None

        self.tg_model = None


    def connect(self, upstream_layer):
        upstream_layer.downstream_layers.append(self)
        self.upstream_layer = upstream_layer
        self.weights = np.empty((np.prod(upstream_layer.shape), self.units), dtype=np.float64)
        self.shape = (self.units, )


    def set_weights(self, weights):
        self.weights[:] = weights


    def get_weights(self):
        return self.weights.copy()


    def compile(self, tg_model):
        self.tg_model = tg_model

        post_nrn_n = np.prod(self.shape)
        for batch_i in range(tg_model.batch_size):

            # Add neuron population
            post_nrn_name = '{}_nrn_{}'.format(self.name, batch_i)
            post_nrn = tg_model.g_model.add_neuron_population(
                post_nrn_name, post_nrn_n, self.model, self.params, self.vars_init
            )
            for gp in self.global_params:
                post_nrn.set_extra_global_param(gp, self.global_params[gp])

            pre_nrn_name = '{}_nrn_{}'.format(self.upstream_layer.name, batch_i)
            syn_name = '{}_to_{}_syn_{}'.format(self.upstream_layer.name, self.name, batch_i)

            # Batch master synapses
            if not tg_model.share_weights or batch_i == 0:
                syn = tg_model.g_model.add_synapse_population(
                    syn_name, 'DENSE_INDIVIDUALG', NO_DELAY, pre_nrn_name, post_nrn_name,
                    'StaticPulse', {}, {'g': self.weights.flatten()}, {}, {}, 'DeltaCurr', {}, {}
                )

            # Batch slave synapses
            else:
                master_syn_name = '{}_to_{}_syn_0'.format(self.upstream_layer.name, self.name)
                syn = tg_model.g_model.add_slave_synapse_population(
                    syn_name, master_syn_name, NO_DELAY, pre_nrn_name, post_nrn_name, 'DeltaCurr', {}, {}
                )


class IFDense(Dense):

    def __init__(self, name, units, threshold=1.0):
        super(IFDense, self).__init__(
            if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold},
            name, units
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
