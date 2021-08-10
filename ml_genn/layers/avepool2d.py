from ml_genn.layers import ConnectivityType, PadMode
from ml_genn.layers import Layer, AvePool2DSynapses
from ml_genn.layers.if_neurons import IFNeurons
from ml_genn.layers.helper import _get_param_2d

class AvePool2D(Layer):

    def __init__(self, name, pool_size, pool_strides=None,
                 pool_padding='valid', connectivity_type='procedural', neurons=IFNeurons()):
        super(AvePool2D, self).__init__(name, neurons)
        self.pool_size = _get_param_2d('pool_size', pool_size)
        self.pool_strides = _get_param_2d('pool_strides', pool_strides, default=self.pool_size)
        self.pool_padding = PadMode(pool_padding)
        self.connectivity_type = ConnectivityType(connectivity_type)

    def connect(self, sources):
        synapses = [
            AvePool2DSynapses(self.pool_size, self.pool_strides, self.pool_padding,
                                   self.connectivity_type) for i in range(len(sources))]
        super(AvePool2D, self).connect(sources, synapses)
