from __future__ import annotations

import numpy as np

from typing import Optional, TYPE_CHECKING
from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Population

class AdaptiveLeakyIntegrateFire(Neuron):
    """A leaky-integrate and fire neuron with an adaptive firing threshold
    as described by [Bellec2018]_.
    
    Args:
        v_thresh:                   Membrane voltage firing threshold
        v_reset:                    After a spike is emitted, this value is
                                    *subtracted* from the membrane voltage
                                    ``v`` if ``relative_reset`` is ``True``.
                                    Otherwise, if ``relative_reset`` is 
                                    ``False``, the membrane voltage is set to
                                    this value.
        v:                          Initial value of membrane voltage
        a:                          Initial value of adaptation
        beta:                       Strength of adaptation
        tau_mem:                    Time constant of membrane voltage [ms]
        tau_adapt:                  Time constant of adaptation [ms]
        relative_reset:             How is ``v`` reset after a spike?
        readout:                    Type of readout to attach to this
                                    neuron's output variable
    """
    
    v_thresh = ValueDescriptor()
    v_reset = ValueDescriptor()
    v = ValueDescriptor()
    a = ValueDescriptor()
    beta = ValueDescriptor()
    tau_mem = ValueDescriptor()
    tau_adapt = ValueDescriptor()
    
    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, a: InitValue = 0.0, beta: InitValue = 0.0174,
                 tau_mem: InitValue = 20.0, tau_adapt: InitValue = 2000.0, 
                 relative_reset: bool = False, readout=None):
        super(AdaptiveLeakyIntegrateFire, self).__init__(readout)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v
        self.a = a
        self.beta = beta
        self.tau_mem = tau_mem
        self.tau_adapt = tau_adapt
        self.relative_reset = relative_reset

    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        # Build basic model
        v_jump = ("v - (v_thresh - v_reset)" if self.relative_reset
                  else "v_reset")
        genn_model = {
            "vars": {"v": ("(-v + i) / tau_mem", v_jump),
                     "a": ("-a / tau_adapt", "a + 1")},
            "threshold": "v - (v_thresh + (beta * a))"}

        # Return model
        return AutoNeuronModel.from_val_descriptors(genn_model, "v", self)
