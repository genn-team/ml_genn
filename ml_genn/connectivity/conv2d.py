from ml_genn.layers import ConnectivityType, PadMode
from ml_genn.layers import Layer, Conv2DSynapses
from ml_genn.layers.if_neurons import IFNeurons
from ml_genn.layers.helper import _get_param_2d

class Conv2D(Layer):

    def __init__(self, name, filters, conv_size, conv_strides=None,
                 conv_padding='valid', connectivity_type='procedural', neurons=IFNeurons()):
        super(Conv2D, self).__init__(name, neurons)
        self.filters = filters
        self.conv_size = _get_param_2d('conv_size', conv_size)
        self.conv_strides = _get_param_2d('conv_strides', conv_strides, default=(1, 1))
        self.conv_padding = PadMode(conv_padding)
        self.connectivity_type = ConnectivityType(connectivity_type)

    def connect(self, sources):
        synapses = [
            Conv2DSynapses(self.filters, self.conv_size, self.conv_strides,
                           self.conv_padding, self.connectivity_type) for i in range(len(sources))]
        super(Conv2D, self).connect(sources, synapses)
