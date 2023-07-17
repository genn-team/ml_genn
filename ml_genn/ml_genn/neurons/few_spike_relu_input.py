from .input import InputBase
from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.snippet import ConstantValueDescriptor

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
    "param_name_types": [("K", "int"), ("Scale", "scalar")],
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
    k = ConstantValueDescriptor()
    alpha = ConstantValueDescriptor()

    def __init__(self, k: int = 10, alpha: float = 25, signed_input=False):
        super(FewSpikeReluInput, self).__init__(var_name="V")

        self.k = k
        self.alpha = alpha
        self.signed_input = signed_input

    def get_model(self, population, dt, batch_size):
        # Calculate scale
        if self.signed_input:
            scale = self.alpha * 2**(-self.k // 2)
        else:
            scale = self.alpha * 2**(-self.k)

        model = genn_model_signed if self.signed_input else genn_model
        return self.create_input_model(
            NeuronModel(model, None, {"K": self.k, "Scale": scale}, {"V": 0.0}),
            batch_size, population.shape)

