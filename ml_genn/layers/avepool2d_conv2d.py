from ml_genn.layers import ConnectivityType, PadMode
from ml_genn.layers import Layer, AvePool2DConv2DSynapses
from ml_genn.layers.if_neurons import IFNeurons
from ml_genn.layers.helper import _get_param_2d

class AvePool2DConv2D(Layer):

    def __init__(self, name, filters, pool_size, conv_size,
                 pool_strides=None, conv_strides=None, pool_padding='valid',
                 conv_padding='valid', connectivity_type='procedural', neurons=IFNeurons()):
        super(AvePool2DConv2D, self).__init__(name, neurons)
        self.filters = filters
        self.pool_size = _get_param_2d('pool_size', pool_size)
        self.conv_size = _get_param_2d('conv_size', conv_size)
        self.pool_strides = _get_param_2d('pool_strides', pool_strides, default=self.pool_size)
        self.conv_strides = _get_param_2d('conv_strides', conv_strides, default=(1, 1))
        self.pool_padding = PadMode(pool_padding)
        self.conv_padding = PadMode(conv_padding)
        self.connectivity_type = ConnectivityType(connectivity_type)

    def connect(self, sources):
        synapses = [
            AvePool2DConv2DSynapses(self.filters, self.pool_size, self.conv_size,
                                    self.pool_strides, self.conv_strides, self.pool_padding,
                                    self.conv_padding, self.connectivity_type) for i in range(len(sources))]
        super(AvePool2DConv2D, self).connect(sources, synapses)
