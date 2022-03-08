from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from .neuron import Neuron, Model
from ..utils import InitValue, Value

genn_model = {
    "var_name_types": [("Input", "scalar", VarAccess_READ_ONLY_DUPLICATE), 
                       ("V", "scalar")],
    "param_name_types": [("Vthr", "scalar")],
    "sim_code":
        """
        $(V) += $(Input) * DT;
        """,
    "threshold_condition_code":
        """
        $(V) >= 1.0
        """,
    "reset_code":
        """
        $(V) = 0.0;
        """,
    "is_auto_refractory_required": False}

class IntegrateFireInput(Neuron):
    def __init__(self, threshold=1.0, v=0.0):
        super(IntegrateFireInput, self).__init__()
        
        self.threshold = Value(threshold)
        self.v = Value(v)

    def get_model(self, population, dt):
        return Model(genn_model, {"Vthr": self.threshold}, {"V": self.v})
