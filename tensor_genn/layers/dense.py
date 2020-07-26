from tensor_genn.layers.dense_connection import DenseConnection
from tensor_genn.layers.layer import Layer
from tensor_genn.layers.neuron_models import if_model


class Dense(Layer):

    def __init__(self, name, model, params, vars_init, global_params,
                 units):
        super(Dense, self).__init__(name, model, params, vars_init, global_params)
        self.units = units


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
