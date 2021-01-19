
class BaseLayer(object):

    def __init__(self, name, neurons):
        self.name = name
        self.neurons = neurons
        self.neurons.name = '{}_nrn'.format(self.name)
        self.downstream_synapses = []
        self.upstream_synapses = []

    def compile_neurons(self, mlg_model):
        self.neurons.compile(mlg_model)

    def compile_synapses(self, mlg_model):
        for synapse in self.upstream_synapses:
            synapse.compile(mlg_model)

    @property
    def shape(self):
        return self.neurons.shape
