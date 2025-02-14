from __future__ import annotations

from typing import TYPE_CHECKING
from .input import InputBase
from .leaky_integrate_fire import LeakyIntegrateFire
from ..utils.model import NeuronModel
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Population

class LeakyIntegrateFireInput(LeakyIntegrateFire, InputBase):
    """A leaky-integrate and fire input neuron.
    
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
        tau_refrac:                 Duration of refractory period [ms]
        relative_reset:             How is ``v`` reset after a spike?
        integrate_during_refrac:    Should ``v`` continue to integrate inputs
                                    during refractory period?
        input_frames:               How many frames does each input have?
        input_frame_timesteps:      How many timesteps should each frame of 
                                    input be presented for?
    """
    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, tau_mem: InitValue = 20.0,
                 tau_refrac: InitValue = None, relative_reset: bool = True,
                 integrate_during_refrac: bool = True, scale_i: bool = False,
                 input_frames=1, input_frame_timesteps=1):
        super(LeakyIntegrateFireInput, self).__init__(
            v_thresh=v_thresh, v_reset=v_reset, v=v, tau_mem=tau_mem, 
            tau_refrac=tau_refrac, relative_reset=relative_reset, 
            integrate_during_refrac=integrate_during_refrac, scale_i=scale_i,
            egp_name="Input", input_frames=input_frames,
            input_frame_timesteps=input_frame_timesteps)


    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        # Get standard integrate-and-fire model
        nm = super(LeakyIntegrateFireInput, self).get_model(population, dt,
                                                            batch_size)

        # Add input logic and replace isyn with input
        return self.create_input_model(nm, batch_size, population.shape,
                                       replace_input="Isyn")
