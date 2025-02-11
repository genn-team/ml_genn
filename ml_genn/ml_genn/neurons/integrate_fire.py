from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING
from .neuron import Neuron
from ..utils.auto_model import AutoNeuronModel
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Population

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
    v_thresh = ValueDescriptor()
    v_reset = ValueDescriptor()
    v = ValueDescriptor()

    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, readout=None, **kwargs):
        super(IntegrateFire, self).__init__(readout, **kwargs)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v

    def get_model(self, population: Population, dt: float,
                  batch_size: int) -> Union[AutoNeuronModel, NeuronModel]:
        genn_model = {
            "vars": {"v": (None, "v_reset")},
            "threshold": "v - v_thresh"}

        return AutoNeuronModel.from_val_descriptors(genn_model, "v", self)
