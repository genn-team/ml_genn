from weakref import proxy

class BaseSynapses(object):

    def __init__(self):
        self.source = None
        self.target = None
        self.output_shape = None
        self.weights = None
        self.tg_model = None
        self.syn = None

    def compile(self, tg_model):
        self.tg_model = proxy(tg_model)
        self.syn = [None] * tg_model.batch_size

    def connect(self, source, target):
        self.source = proxy(source)
        self.target = proxy(target)
        source.downstream_synapses.append(self)
        target.upstream_synapses.append(self)

    def set_weights(self, weights):
        self.weights[:] = weights

    def get_weights(self):
        return self.weights.copy()
