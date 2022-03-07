from . import Synapse

genn_model = {
    "apply_input_code":
        """
        $(Isyn) += $(inSyn);
        """,
    "decay_code":
        """
        $(inSyn) = 0;
        """}
        
class Delta(Synapse):
    def __init__(self):
        super(Delta, self).__init__()

    def get_model(self, population):
        return genn_model

    def get_param_vals(self, dt):
        return {}

    @property
    def var_vals(self):
        return {}
