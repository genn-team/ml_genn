from . import Synapse

class Delta(Synapse):
    def __init__(self):
        super(Delta, self).__init__()

    def get_model(self, population):
        return "DeltaCurr"

    @property
    def params(self):
        return {}

    @property
    def vars(self):
        return {}