from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Population


genn_model = {
    "vars": [("V", "scalar")],
    "params": [("Vthresh", "scalar"), ("Vreset", "scalar")],
    "sim_code":
        """
        V += Isyn;
        """,
    "threshold_condition_code":
        """
        V >= Vthresh
        """,
    "reset_code":
        """
        V = Vreset;
        """}


class IntegrateFire(Neuron):
    """An integrate and fire neuron.
    
    Args:
        v_thresh:   Membrane voltage firing threshold
        v_reset:    After a spike is emitted, the membrane 
                    voltage is set to this value.
        v:          Initial value of membrane voltage
        readout:    Type of readout to attach to this
                    neuron's output variable
    """
    v_thresh = ValueDescriptor("Vthresh")
    v_reset = ValueDescriptor("Vreset")
    v = ValueDescriptor("V")

    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, readout=None, **kwargs):
        super(IntegrateFire, self).__init__(readout, **kwargs)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v

    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        return NeuronModel.from_val_descriptors(genn_model, "V", self, dt)
