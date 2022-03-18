from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from .input_base import InputBase
from .neuron import Neuron
from ..utils import InitValue, NeuronModel, Value

genn_model = {
    "param_name_types": [("K", "int"), ("Scale", "scalar")],
    "var_name_types": [("V", "scalar")],
    "sim_code":
        """
        // Convert K to integer
        const int kInt = (int)$(K);
        
        // Get timestep within presentation
        const int pipeTimestep = (int)($(t) / DT);

        const scalar hT = $(Scale) * (1 << (kInt - (1 + pipeTimestep)));
        """,
    "threshold_condition_code":
        """
        $(V) >= hT
        """,
    "reset_code":
        """
        $(V) -= hT;
        """,
    "is_auto_refractory_required": False}

genn_model_signed = {
    "param_name_types": ["K", "Scale"],    
    "var_name_types": [("V", "scalar")],
    "sim_code":
        """
        // Convert K to integer
        const int halfK = (int)$(K) / 2;

        // Get timestep within presentation
        const int pipeTimestep = (int)($(t) / DT);

        // Split timestep into interleaved positive and negative
        const int halfPipetimestep = pipeTimestep / 2;
        const bool positive = (pipeTimestep % 2) == 0;
        const scalar hT = $(Scale) * (1 << (halfK - (1 + halfPipetimestep)));
        """,
    "threshold_condition_code":
        """
        (positive && $(V) >= hT) || (!positive && $(V) < -hT)
        """,
    "reset_code":
        """
        if(positive) {
            $(V) -= hT;
        }
        else {
            $(V) += hT;
        }
        """,
    "is_auto_refractory_required": False}
    

class FewSpikeReluInput(Neuron, InputBase):
    def __init__(self, k=10, alpha=25, signed_input=False):
        super(FewSpikeReluInput, self).__init__(var_name="V")

        self.k = Value(k)
        self.alpha = Value(alpha)
        self.signed_input = signed_input
        
        if self.k.is_initializer or self.alpha.is_initializer:
            raise NotImplementedError("FewSpike ReLU input model does not "
                                      "currently support k or alpha values "
                                      "specified using Initialiser objects")

    def get_model(self, population, dt):
        # Calculate scale
        if self.signed_input:
            scale = self.alpha.value * 2**(-self.k.value // 2)
        else:
            scale = self.alpha.value * pars[1] * 2**(-self.k.value)
        
        model = genn_model_signed if self.signed_input else genn_model,
        return NeuronModel(model, {"K": self.k, "Scale": Value(scale)},
                           {"Input": Value(0.0), "V": Value(0.0)})
