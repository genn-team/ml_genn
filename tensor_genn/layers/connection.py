import numpy as np
from enum import Enum


class PadMode(Enum):
    VALID = 'valid'
    SAME = 'same'


class BaseConnection(object):

    def __init__(self):
        self.source = None
        self.target = None
        self.weights = None
        self.tg_model = None
        self.syn = None


    def connect(self, source, target):
        self.source = source
        self.target = target
        source.downstream_connections.append(self)
        target.upstream_connections.append(self)


    def set_weights(self, weights):
        self.weights[:] = weights


    def get_weights(self):
        return self.weights.copy()


    def compile(self, tg_model):
        self.tg_model = tg_model
        self.syn = [None] * tg_model.batch_size


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
