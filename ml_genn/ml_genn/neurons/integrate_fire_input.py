from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from .input_base import InputBase
from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

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
    v_thresh = ValueDescriptor
    v_reset = ValueDescriptor
    v = ValueDescriptor
    
    def __init__(self, v_thresh:InitValue=1.0, v_reset:InitValue=0.0, 
                 v:InitValue=0.0):
        super(IntegrateFireInput, self).__init__()
        
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v

    def get_model(self, population, dt):
        return NeuronModel(genn_model, 
                           {"Vthresh": self.v_thresh, "Vreset": self.v_reset}, 
                           {"V": self.v, "Input": 0.0})
