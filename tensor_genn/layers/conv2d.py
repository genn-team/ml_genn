from tensor_genn.layers import SynapseType, PadMode
from tensor_genn.layers import Layer, Conv2DSynapses

class Conv2D(Layer):

    def __init__(self, name, filters, conv_size, conv_strides=None,
                 conv_padding='valid', synapse_type='procedural', neurons=None):
        super(Conv2D, self).__init__(name, neurons)
        self.filters = filters
        self.conv_size = conv_size
        if conv_strides == None:
            self.conv_strides = (1, 1)
        else:
            self.conv_strides = conv_strides
        self.conv_padding = PadMode(conv_padding)
        self.synapse_type = SynapseType(synapse_type)

    def connect(self, sources):
        synapses = [
            Conv2DSynapses(self.filters, self.conv_size, self.conv_strides,
                           self.conv_padding, self.synapse_type) for i in range(len(sources))]
        super(Conv2D, self).connect(sources, synapses)
