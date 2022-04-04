import numpy as np

from .synapse import Synapse
from ..utils import InitValue, SynapseModel, ValueDescriptor

from ..utils import is_value_initializer

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
    tau = ValueDescriptor()
    
    def __init__(self, tau:InitValue=5.0):
        super(Exponential, self).__init__()
        
        self.tau = tau
        
        if is_value_initializer(self.tau):
            raise NotImplementedError("Exponential synapse model does not "
                                      "currently support tau values specified"
                                      " using Initialiser objects")

    def get_model(self, connection, dt):
        return SynapseModel(genn_model, 
                            {"ExpDecay": np.exp(-dt / self.tau.value)}, {})
