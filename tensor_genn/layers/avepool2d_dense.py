from tensor_genn.layers import SynapseType, PadMode
from tensor_genn.layers import Layer, AvePool2DDenseSynapse
from tensor_genn.layers.neuron_models import if_model


class AvePool2DDense(Layer):

    def __init__(self, model, params, vars_init, global_params, name, 
                 units, pool_size, pool_strides=None, pool_padding='valid', 
                 synapse_type='procedural', signed_spikes=False):
        super(AvePool2DDense, self).__init__(model, params, vars_init, 
                                             global_params, name, signed_spikes)
        self.units = units
        self.pool_size = pool_size
        if pool_strides == None:
            self.pool_strides = (pool_size[0], pool_size[1])
        else:
            self.pool_strides = pool_strides
        self.pool_padding = PadMode(pool_padding)
        self.synapse_type = SynapseType(synapse_type)

    def connect(self, sources):
        synapses = [
            AvePool2DDenseSynapse(self.units, self.pool_size, 
                                  self.pool_strides, self.pool_padding,
                                  self.synapse_type) for i in range(len(sources))]
        super(AvePool2DDense, self).connect(sources, synapses)


class IFAvePool2DDense(AvePool2DDense):

    def __init__(self, name, units, pool_size, pool_strides=None, 
                 pool_padding='valid', synapse_type='procedural', 
                 threshold=1.0, signed_spikes=False):
        super(IFAvePool2DDense, self).__init__(
            if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold},
            name, units, pool_size, pool_strides, pool_padding, 
            synapse_type, signed_spikes)

    def set_threshold(self, threshold):
        self.global_params['Vthr'] = threshold

        if self.nrn is not None:
            for batch_i in range(self.tg_model.batch_size):
                nrn = self.nrn[batch_i]
                nrn.extra_global_params['Vthr'].view[:] = threshold
