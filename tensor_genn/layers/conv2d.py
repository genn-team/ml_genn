from tensor_genn.layers.base_connection import PadMode
from tensor_genn.layers.conv2d_connection import Conv2DConnection
from tensor_genn.layers.layer import Layer
from tensor_genn.layers.neuron_models import if_model


class Conv2D(Layer):

    def __init__(self, name, model, params, vars_init, global_params,
                 filters, conv_size, conv_strides=None, conv_padding='valid'):
        super(Conv2D, self).__init__(name, model, params, vars_init, global_params)
        self.filters = filters
        self.conv_size = conv_size
        if conv_strides == None:
            self.conv_strides = (1, 1)
        else:
            self.conv_strides = conv_strides
        self.conv_padding = PadMode(conv_padding)


    def connect(self, sources):
        connections = [
            Conv2DConnection(self.filters, self.conv_size, self.conv_strides, self.conv_padding)
            for i in range(len(sources))
        ]
        super(Conv2D, self).connect(sources, connections)


class IFConv2D(Conv2D):

    def __init__(self, name, filters, conv_size, conv_strides=None, conv_padding='valid', threshold=1.0):
        super(IFConv2D, self).__init__(
            name, if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold},
            filters, conv_size, conv_strides, conv_padding
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
