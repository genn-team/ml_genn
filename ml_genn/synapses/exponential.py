import numpy as np

from . import Synapse
from ..initializers import Initializer
from ..utils import InitValue, Value

genn_model = {
    "param_name_types": [("ExpDecay", "scalar")],
    "apply_input_code":
        """
        $(Isyn) += $(inSyn);
        """,
    "decay_code":
        """
        $(inSyn) *= $(ExpDecay);
        """}
        
class Exponential(Synapse):
    def __init__(self, tau=5.0):
        super(Exponential, self).__init__()
        self.tau = Value(tau)

    def get_model(self, population):
        return genn_model

    def get_param_vals(self, dt):
        if isinstance(self.tau, Initializer):
            raise NotImplementedError("Exponential synapse model does not "
                                      "currently support tau values specified"
                                      " using Initialiser objects")
        # Calculate ExpDecay parameter from tau
        return {"ExpDecay": Value(np.exp(-dt / self.tau.value))}

    @property
    def var_vals(self):
        return {}