from __future__ import annotations

import numpy as np

from typing import Optional, Union, TYPE_CHECKING
from .neuron import Neuron
from ..utils.auto_model import AutoNeuronModel
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Population

class LeakyIntegrateFire(Neuron):
    """A leaky-integrate and fire neuron.
    
    Args:
        v_thresh:                   Membrane voltage firing threshold
        v_reset:                    After a spike is emitted, this value is
                                    *subtracted* from the membrane voltage
                                    ``v`` if ``relative_reset`` is ``True``.
                                    Otherwise, if ``relative_reset`` is 
                                    ``False``, the membrane voltage is set to
                                    this value.
        v:                          Initial value of membrane voltage
        tau_mem:                    Time constant of membrane voltage [ms]
        relative_reset:             How is ``v`` reset after a spike?
        readout:                    Type of readout to attach to this
                                    neuron's output variable
    """
    
    v_thresh = ValueDescriptor()
    v_reset = ValueDescriptor()
    v = ValueDescriptor()
    tau_mem = ValueDescriptor()

    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, tau_mem: InitValue = 20.0,
                 relative_reset: bool = True, readout=None, **kwargs):
        super(LeakyIntegrateFire, self).__init__(readout, **kwargs)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v
        self.tau_mem = tau_mem
        self.relative_reset = relative_reset

    def get_model(self, population: Population, dt: float,
                  batch_size: int) -> Union[AutoNeuronModel, NeuronModel]:
        # Build basic model
        v_jump = ("v - (v_thresh - v_reset)" if self.relative_reset
                  else "v_reset")
        genn_model = {
            "vars": {"v": ("(-v + Isyn) / tau_mem", v_jump)},
            "threshold": "v - v_thresh"}

        # Return model
        return AutoNeuronModel.from_val_descriptors(genn_model, "v", self)
