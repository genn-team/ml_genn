from tensor_genn.layers.base_connection import PadMode
from tensor_genn.layers.avepool2d_conv2d_connection import AvePool2DConv2DConnection
from tensor_genn.layers.layer import Layer
from tensor_genn.layers.neuron_models import if_model


class AvePool2DConv2D(Layer):

    def __init__(self, name, model, params, vars_init, global_params,
                 filters, pool_size, conv_size, pool_strides=None, conv_strides=None,
                 pool_padding='valid', conv_padding='valid'):
        super(AvePool2DConv2D, self).__init__(name, model, params, vars_init, global_params)
        self.filters = filters
        self.pool_size = pool_size
        self.conv_size = conv_size
        if pool_strides == None:
            self.pool_strides = (pool_size[0], pool_size[1])
        else:
            self.pool_strides = pool_strides
        if conv_strides == None:
            self.conv_strides = (1, 1)
        else:
            self.conv_strides = conv_strides
        self.pool_padding = PadMode(pool_padding)
        self.conv_padding = PadMode(conv_padding)


    def connect(self, sources):
        connections = [
            AvePool2DConv2DConnection(self.filters, self.pool_size, self.conv_size, self.pool_strides,
                                      self.conv_strides, self.pool_padding, self.conv_padding)
            for i in range(len(sources))
        ]
        super(AvePool2DConv2D, self).connect(sources, connections)


class IFAvePool2DConv2D(AvePool2DConv2D):

    def __init__(self, name, filters, pool_size, conv_size, pool_strides=None, conv_strides=None,
                 pool_padding='valid', conv_padding='valid', threshold=1.0):
        super(IFAvePool2DConv2D, self).__init__(
            name, if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold},
            filters, pool_size, conv_size, pool_strides, conv_strides, pool_padding, conv_padding
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
