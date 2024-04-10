from __future__ import annotations

from typing import TYPE_CHECKING
from .input import InputBase
from .integrate_fire import IntegrateFire
from ..utils.model import NeuronModel
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Population


class IntegrateFireInput(IntegrateFire, InputBase):
    """An integrate and fire input neuron.
    
    Args:
        v_thresh:               Membrane voltage firing threshold
        v_reset:                After a spike is emitted, the membrane
                                voltage is set to this value.
        v:                      Initial value of membrane voltage
        input_frames:           How many frames does each input have?
        input_frame_timesteps:  How many timesteps should each frame of 
                                input be presented for?
    """
    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, input_frames=1, input_frame_timesteps=1):
        super(IntegrateFireInput, self).__init__(
            v_thresh=v_thresh, v_reset=v_reset, v=v,
            egp_name="Input", input_frames=input_frames,
            input_frame_timesteps=input_frame_timesteps)


    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        # Get standard integrate-and-fire model
        nm = super(IntegrateFireInput, self).get_model(population, dt,
                                                       batch_size)

        # Add input logic and replace isyn with input
        return self.create_input_model(nm, batch_size, population.shape,
                                       replace_input="Isyn")
