from tensor_genn.layers.base_layer import BaseLayer


class Layer(BaseLayer):

    def __init__(self, model, params, vars_init, 
                 global_params, name, signed_spikes=False):
        super(Layer, self).__init__(model, params, vars_init, 
                                    global_params, name, signed_spikes)

    def compile(self, tg_model):
        super(Layer, self).compile(tg_model)

        for synapse in self.upstream_synapses:
            synapse.compile(tg_model)

    def connect(self, sources, synapses):
        if len(sources) != len(synapses):
            raise ValueError('sources list and synapse list length mismatch')

        for source, synapse in zip(sources, synapses):
            synapse.connect(source, self)

    def set_weights(self, weights):
        if len(weights) != len(self.upstream_synapses):
            raise ValueError('weight matrix list and upsteam synapse list length mismatch')

        for synapse, w in zip(self.upstream_synapses, weights):
            synapse.set_weights(w)

    def get_weights(self):
        return [synapse.get_weights() for synapse in self.upstream_synapses]


class IFLayer(Layer):

    def __init__(self, name, threshold=1.0, signed_spikes=False):
        super(IFLayer, self).__init__(
            if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold},
            name, signed_spikes)

    def set_threshold(self, threshold):
        self.global_params['Vthr'] = threshold

        if self.nrn is not None:
            for batch_i in range(self.tg_model.batch_size):
                nrn = self.nrn[batch_i]
                nrn.extra_global_params['Vthr'].view[:] = threshold
