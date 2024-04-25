from __future__ import annotations

from typing import TYPE_CHECKING
from .input import InputBase
from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.snippet import ConstantValueDescriptor

if TYPE_CHECKING:
    from .. import Population


genn_model = {
    "params": [("K", "int"), ("Scale", "scalar")],
    "vars": [("V", "scalar")],
    "sim_code":
        """
        // Convert K to integer
        const int kInt = (int)K;

        // Get timestep within presentation
        const int pipeTimestep = (int)(t / dt);

        const scalar hT = Scale * (1 << (kInt - (1 + pipeTimestep)));
        """,
    "threshold_condition_code":
        """
        V >= hT
        """,
    "reset_code":
        """
        V -= hT;
        """}

genn_model_signed = {
    "params": [("K", "int"), ("Scale", "scalar")],
    "vars": [("V", "scalar")],
    "sim_code":
        """
        // Convert K to integer
        const int halfK = (int)K / 2;

        // Get timestep within presentation
        const int pipeTimestep = (int)(t / dt);

        // Split timestep into interleaved positive and negative
        const int halfPipetimestep = pipeTimestep / 2;
        const bool positive = (pipeTimestep % 2) == 0;
        const scalar hT = Scale * (1 << (halfK - (1 + halfPipetimestep)));
        """,
    "threshold_condition_code":
        """
        (positive && V >= hT) || (!positive && V < -hT)
        """,
    "reset_code":
        """
        if(positive) {
            V -= hT;
        }
        else {
            V += hT;
        }
        """}


class FewSpikeReluInput(Neuron, InputBase):
    """A few-spike neuron to encode inputs using a 
    ReLU activation as described by [Stockl2021]_.
    
    Should typically be created by converting an ANN to an SNN using
    :class:`ml_genn_tf.converters.FewSpike`.
    
    Args:
        k:              Number of timesteps to encode activation over.
        alpha:          Scaling factor to apply to activations.
        signed_input:   Are inputs expected to be both positive and negative?
    """
    
    k = ConstantValueDescriptor()
    alpha = ConstantValueDescriptor()

    def __init__(self, k: int = 10, alpha: float = 25, signed_input=False):
        super(FewSpikeReluInput, self).__init__(var_name="V")

        self.k = k
        self.alpha = alpha
        self.signed_input = signed_input

    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        # Calculate scale
        if self.signed_input:
            scale = self.alpha * 2**(-self.k // 2)
        else:
            scale = self.alpha * 2**(-self.k)

        # Return appropriate neuron model
        # **NOTE** because this model doesn't support time-varying input
        # and input is read into an existing state variable, no need to use create_input_model
        model = genn_model_signed if self.signed_input else genn_model
        return NeuronModel(model, None, {"K": self.k, "Scale": scale}, {"V": 0.0})

