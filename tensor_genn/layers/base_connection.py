from enum import Enum


class PadMode(Enum):
    VALID = 'valid'
    SAME = 'same'


class BaseConnection(object):

    def __init__(self):
        self.source = None
        self.target = None
        self.output_shape = None
        self.weights = None
        self.tg_model = None
        self.syn = None


    def compile(self, tg_model):
        self.tg_model = tg_model
        self.syn = [None] * tg_model.batch_size


    def connect(self, source, target):
        self.source = source
        self.target = target
        source.downstream_connections.append(self)
        target.upstream_connections.append(self)


    def set_weights(self, weights):
        self.weights[:] = weights


    def get_weights(self):
        return self.weights.copy()
