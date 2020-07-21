import numpy as np

from pygenn.genn_wrapper import NO_DELAY

from tensor_genn.layers import BaseConnection
from tensor_genn.layers import Layer
from tensor_genn.genn_models import if_model


class DenseConnection(BaseConnection):

    def __init__(self, units):
        super(DenseConnection, self).__init__()
        self.units = units


    def connect(self, source, target):
        super(DenseConnection, self).connect(source, target)

        shape = (self.units, )
        if target.shape is None:
            target.shape = shape
        elif target.shape != shape:
            raise RuntimeError('layer shape mismatch')

        self.weights = np.empty((np.prod(source.shape), self.units), dtype=np.float64)


    def compile(self, tg_model):
        super(DenseConnection, self).compile(tg_model)

        for batch_i in range(tg_model.batch_size):
            pre_nrn = self.source.nrn[batch_i]
            post_nrn = self.target.nrn[batch_i]
            syn_name = '{}_to_{}_syn_{}'.format(self.source.name, self.target.name, batch_i)

            # Batch master synapses
            if not tg_model.share_weights or batch_i == 0:
                self.syn[batch_i] = tg_model.g_model.add_synapse_population(
                    syn_name, 'DENSE_INDIVIDUALG', NO_DELAY, pre_nrn, post_nrn,
                    'StaticPulse', {}, {'g': self.weights.flatten()}, {}, {}, 'DeltaCurr', {}, {}
                )

            # Batch slave synapses
            else:
                master_syn_name = '{}_to_{}_syn_0'.format(self.source.name, self.target.name)
                self.syn[batch_i] = tg_model.g_model.add_slave_synapse_population(
                    syn_name, master_syn_name, NO_DELAY, pre_nrn, post_nrn, 'DeltaCurr', {}, {}
                )


class Dense(Layer):

    def __init__(self, name, model, params, vars_init, global_params,
                 units):
        super(Dense, self).__init__(name, model, params, vars_init, global_params)
        self.units = units
        self.weights = None


    def connect(self, sources):
        connections = [
            DenseConnection(self.units)
            for i in range(len(sources))
        ]
        super(Dense, self).connect(sources, connections)


class IFDense(Dense):

    def __init__(self, name, units, threshold=1.0):
        super(IFDense, self).__init__(
            name, if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold},
            units
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
