from ml_genn.layers import ConnectivityType, PadMode
from ml_genn.layers import Layer, AvePool2DDenseSynapses
from ml_genn.layers.if_neurons import IFNeurons

class AvePool2DDense(Layer):

    def __init__(self, name, units, pool_size, pool_strides=None,
                 pool_padding='valid', connectivity_type='procedural', neurons=IFNeurons()):
        super(AvePool2DDense, self).__init__(name, neurons)
        self.units = units
        self.pool_size = pool_size
        if pool_strides == None:
            self.pool_strides = (pool_size[0], pool_size[1])
        else:
            self.pool_strides = pool_strides
        self.pool_padding = PadMode(pool_padding)
        self.connectivity_type = ConnectivityType(connectivity_type)

    def connect(self, sources):
        synapses = [
            AvePool2DDenseSynapses(self.units, self.pool_size, 
                                   self.pool_strides, self.pool_padding,
                                   self.connectivity_type) for i in range(len(sources))]
        super(AvePool2DDense, self).connect(sources, synapses)
