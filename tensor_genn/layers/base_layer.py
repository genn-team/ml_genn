
class BaseLayer(object):

    def __init__(self, name, neurons):
        self.name = name
        self.neurons = neurons
        self.neurons.name = '{}_nrn'.format(self.name)
        self.downstream_synapses = []
        self.upstream_synapses = []

    def compile_neurons(self, tg_model):
        self.neurons.compile(tg_model)

    def compile_synapses(self, tg_model):
        for synapse in self.upstream_synapses:
            synapse.compile(tg_model)

    @property
    def shape(self):
        return self.neurons.shape
