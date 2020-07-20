
class BaseConnection(object):

    def __init__(self):
        self.source = None
        self.target = None
        self.weights = None
        self.tg_model = None


    def connect(self, source, target):
        self.source = source
        self.target = target
        source.downstream_connections.append(self)
        target.upstream_connections.append(self)


    def set_weights(self, weights):
        self.weights[:] = weights


    def get_weights(self):
        return self.weights.copy()


    def compile(self, tg_model):
        self.tg_model = tg_model
