from . import Synapse

class Delta(Synapse):
    def __init__(self):
        super(Delta, self).__init__()

    def get_model(self, population):
        return "DeltaCurr"

    @property
    def param_vals(self):
        return {}

    @property
    def var_vals(self):
        return {}
