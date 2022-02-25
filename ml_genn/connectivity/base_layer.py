
class BaseLayer(object):

    def __init__(self, name, neurons):
        self.name = name
        self.shape = None
        self.neurons = neurons
        self.downstream_synapses = []
        self.upstream_synapses = []

    def compile_neurons(self, mlg_model):
        self.neurons.compile(mlg_model, self)

    def compile_synapses(self, mlg_model):
        for synapse in self.upstream_synapses:
            name = '{}_to_{}_syn'.format(synapse.source().name, synapse.target().name)
            synapse.compile(mlg_model, name)
