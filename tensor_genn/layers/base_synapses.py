from weakref import proxy

class BaseSynapses(object):

    def __init__(self):
        self.name = None
        self.source = None
        self.target = None
        self.weights = None
        self.syn = None

    def connect(self, source, target):
        self.source = proxy(source)
        self.target = proxy(target)
        source.downstream_synapses.append(self)
        target.upstream_synapses.append(self)
        self.name = '{}_to_{}_syn'.format(self.source.name, self.target.name)

    def set_weights(self, weights):
        self.weights[:] = weights

    def get_weights(self):
        return self.weights.copy()

    def compile(self, tg_model):
        self.syn = [None] * tg_model.batch_size
