from __future__ import annotations

from typing import TYPE_CHECKING
from .input import InputBase
from .neuron import Neuron
from ..utils.model import NeuronModel

if TYPE_CHECKING:
    from .. import Population

class BinarySpikeInput(Neuron, InputBase):
    """Input neuron which simply emits a spike if the input is greater
    than zero. Optionally, it can also emit a 'negative' spike if input 
    is less than zero.
    
    Args:
        signed_spikes:          Should negative spikes be emitted
                                if input is less than zero?
        input_frames:           How many frames does each input have?
        input_frame_timesteps:  How many timesteps should each frame of 
                                input be presented for?
    """
    def __init__(self, signed_spikes=False, input_frames=1,
                 input_frame_timesteps=1):
        super(BinarySpikeInput, self).__init__(
            egp_name="Input", input_frames=input_frames,
            input_frame_timesteps=input_frame_timesteps)

        self.signed_spikes = signed_spikes
        if self.signed_spikes and input_frames > 1:
            raise NotImplementedError("Signed spike input cannot currently "
                                      "be used with time-varying inputs ")

    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        genn_model = {
            "sim_code":
                """
                const bool spike = Input != 0.0;
                """,
            "threshold_condition_code":
                """
                Input > 0.0 && spike
                """}

        # If signed spikes are enabled, add negative threshold condition
        if self.signed_spikes:
            genn_model["negative_threshold_condition_code"] =\
                """
                Input_pre < 0.0 && spike
                """

        # Add standard input logic to model and return
        return self.create_input_model(NeuronModel(genn_model, None, {}, {}),
                                       batch_size, population.shape, replace_input="Input")
