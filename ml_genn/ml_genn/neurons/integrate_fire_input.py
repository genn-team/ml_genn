from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from .input_base import InputBase
from .neuron import Neuron, Model
from ..utils import InitValue, Value

genn_model = {
    "var_name_types": [("Input", "scalar", VarAccess_READ_ONLY_DUPLICATE), 
                       ("V", "scalar")],
    "param_name_types": [("Vthresh", "scalar"), ("Vreset", "scalar")],
    "sim_code":
        """
        $(V) += $(Input);
        """,
    "threshold_condition_code":
        """
        $(V) >= $(Vthresh)
        """,
    "reset_code":
        """
        $(V) = $(Vreset);
        """,
    "is_auto_refractory_required": False}

class IntegrateFireInput(Neuron, InputBase):
    def __init__(self, v_thresh=1.0, v_reset=0.0, v=0.0):
        super(IntegrateFireInput, self).__init__()
        
        self.v_thresh = Value(v_thresh)
        self.v_reset = Value(v_reset)
        self.v = Value(v)

    def get_model(self, population, dt):
        return Model(genn_model, 
                     {"Vthresh": self.v_thresh, "Vreset": self.v_reset}, 
                     {"V": self.v})