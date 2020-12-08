from weakref import proxy

class BaseNeurons(object):

    def __init__(self):
        self.shape = None
        self.tg_model = None
        self.nrn = None

    def compile(self, tg_model):
        self.tg_model = proxy(tg_model)
        self.nrn = [None] * tg_model.batch_size
