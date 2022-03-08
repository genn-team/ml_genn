from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from .neuron import Neuron, Model
from ..utils import InitValue, Value

genn_model = {
    "param_name_types": ["K", "Scale"],    
    "var_name_types": [("Input", "scalar", VarAccess_READ_ONLY_DUPLICATE), 
                       ("V", "scalar")],
    "sim_code":
        """
        // Convert K to integer
        const int kInt = (int)$(K);
        
        // Get timestep within presentation
        const int pipeTimestep = (int)($(t) / DT);

        // If this is the first timestep, apply input
        if(pipeTimestep == 0) {
            $(V) = $(Input);
        }
        
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
    "var_name_types": [("Input", "scalar"), ("V", "scalar")],
    "sim_code":
        """
        // Convert K to integer
        const int halfK = (int)$(K) / 2;

        // Get timestep within presentation
        const int pipeTimestep = (int)($(t) / DT);

        // If this is the first timestep, apply input
        if(pipeTimestep == 0) {
            $(V) = $(Input);
        }

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
    

class FewSpikeInput(Neuron):
    def __init__(self, k=10, alpha=25, signed_input=False):
        super(FewSpikeInput, self).__init__()
        self.k = Value(k)
        self.alpha = Value(alpha)
        self.signed_input = signed_input
        
        if self.k.is_initializer or self.alpha.is_initializer:
            raise NotImplementedError("Few spike encoder model does not "
                                      "currently support k or alpha values "
                                      "specified using Initialiser objects")
        
    def get_model(self, population, dt):
        # Calculate scale
        if self.signed_input:
            scale = self.alpha * 2**(-self.k // 2)
        else:
            scale = self.alpha * pars[1] * 2**(-pars[0])
        
        return Model(genn_model_signed if self.signed_input else genn_model, 
                     {"K": self.k, "Scale": scale}, 
                     {"Input": Value(0.0), "V": self.v})
