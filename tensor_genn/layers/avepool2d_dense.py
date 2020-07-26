from tensor_genn.layers.base_connection import PadMode
from tensor_genn.layers.avepool2d_dense_connection import AvePool2DDenseConnection
from tensor_genn.layers.layer import Layer
from tensor_genn.layers.neuron_models import if_model


class AvePool2DDense(Layer):

    def __init__(self, name, model, params, vars_init, global_params,
                 units, pool_size, pool_strides=None, pool_padding='valid'):
        super(AvePool2DDense, self).__init__(name, model, params, vars_init, global_params)
        self.units = units
        self.pool_size = pool_size
        if pool_strides == None:
            self.pool_strides = (pool_size[0], pool_size[1])
        else:
            self.pool_strides = pool_strides
        self.pool_padding = PadMode(pool_padding)


    def connect(self, sources):
        connections = [
            AvePool2DDenseConnection(self.units, self.pool_size, self.pool_strides, self.pool_padding)
            for i in range(len(sources))
        ]
        super(AvePool2DDense, self).connect(sources, connections)


class IFAvePool2DDense(AvePool2DDense):

    def __init__(self, name, units, pool_size, pool_strides=None, pool_padding='valid', threshold=1.0):
        super(IFAvePool2DDense, self).__init__(
            name, if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold},
            units, pool_size, pool_strides, pool_padding
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
