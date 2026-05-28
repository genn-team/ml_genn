from __future__ import annotations

import numpy as np

from collections import namedtuple
from pygenn import VarAccess
from typing import Sequence, Union, TYPE_CHECKING
from .input import Input
from .neuron import Neuron
from ..utils.model import NeuronModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Population

signed_genn_model = {
    "vars": [("SpikeTime", "scalar", VarAccess.READ_ONLY_DUPLICATE),
             ("SpikePolarity", "int", VarAccess.READ_ONLY_DUPLICATE),
             ("Spiked", "uint8_t")],
    "threshold_condition_code":
        """
        (!Spiked) && (t > SpikeTime)
        """,
    "reset_code":
        """
        Spiked = true;
        """
}

genn_model = {
    "vars": [("SpikeTime", "scalar", VarAccess.READ_ONLY_DUPLICATE),],
    "threshold_condition_code":
        """
        (!Spiked) && (t > SpikeTime)
        """,
    "reset_code":
        """
        Spiked = true;
        """
}
    

class LatencyInput(Neuron, Input):
    """An input neuron which emits one spike per trial according to a latency
    code. Can be signed spikes or unsigned.
    
    Args:
        latency_method: "log_latency" or "linear_latency"
        max_time: end of trial
        min_time: first possible spike time
        thresh: threshold for when to emit a spike
        signed: whether to emit signed spikes
    """
    def __init__(self,
                 latency_method: Union["log","linear"] ="linear",
                 max_time: float = 20.0,
                 min_time: float = 0.0,
                 thresh: int = 1,
                 signed: bool = False):
        super().__init__()
        self.latency_method = latency_method
        self.max_time = max_time
        self.min_time = min_time
        self.thresh = thresh
        self.signed = signed

    def set_input(self, genn_pop, batch_size: int, shape,
                  input):
        # expecting input in the form of gray levels -255 to 255 int
        input = np.asarray(input)

        # Get view
        spike_time_var = genn_pop.vars["SpikeTime"]
        if self.signed:
            spike_polarity_var = genn_pop.vars["SpikePolarity"]

        # Split input shape into the bit that should match population
        # shape and the bit that should match time and batch
        input_shape_dims = input.shape[-len(shape):]
        input_batch_dims = input.shape[:-len(shape)]

        # Calculate start and end spike indices
        if input_shape_dims != shape:
            raise RuntimeError(f"Input shape {input.shape} does not match "
                               f"population shape {shape}")

        time_range = self.max_time - self.min_time        
        if self.signed:
            polarity = np.sign(input)
        input = np.abs(input)
        if self.latency_method == "linear":
            spike_time =  (((255.0 - input) / 255.0) * time_range) + self.min_time
            fail = np.where(spike_time < self.min_time)[0]
            if len(fail) > 0:
                print(fail)
        else:
            # scale so that the values are <= range and add min_time
            tau_eff = time_range/np.log(self.thresh+1)
            spike_time = tau_eff * np.log(spike_pixels / (spike_pixels - self.thresh)) + self.min_time
        # set spike times for sub-threshold neurons to much beyond max_time
        # **THINK** is there a better way/ some float_max maybe?
        spike_time[input <= self.thresh] = self.max_time*10.0
        if batch_size == 1:
            # Check input shape either has no batch
            # dimension or it has a length of 1
            if (len(input_batch_dims) != 0
                and (len(input_batch_dims) != 1
                     or input_batch_dims[0] != 1)):
                raise RuntimeError(f"Input shape {input.shape} does "
                                   f"not match batch size {batch_size}")
            # Flatten input and copy into view
            spike_time_var.view[:] = spike_time.flatten()
            if self.signed:
                spike_polarity_var.view[:] = polarity.flatten() 
        # Otherwise
        else:
            # Check input shape has batch dimension
            # and this is less than or equal to batch size
            if (len(input_batch_dims) != 1
                or input_batch_dims[0] > batch_size):
                raise RuntimeError(f"Input shape {input.shape} does "
                                   f"not match batch size {batch_size}")
            # Reshape input into batches of flattened data
            batched_input = np.reshape(spike_time, (-1, np.prod(shape)))
            batched_polarity = np.reshape(polarity, (-1, np.prod(shape)))
            # If we have a full batch
            input_batch_size = batched_input.shape[0]
            if input_batch_size == batch_size:
                spike_time_var.view[:] = batched_input
                if self.signed:
                    spike_polarity_var.view[:] = batched_polarity
           # Otherwise, pad up to full batch
            else:
                spike_time_var.view[:] = np.pad(
                    batched_input,
                    ((0, batch_size - input_batch_size), (0, 0)))
                if self.signed:
                    spike_polarity_var.view[:] = np.pad(
                    batched_polarity,
                    ((0, batch_size - input_batch_size), (0, 0)))
            # Push variable to device
        spike_time_var.push_to_device()
        if self.signed:
            spike_polarity_var.push_to_device()
            
    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        if self.signed:
            return NeuronModel(signed_genn_model, None, {}, 
                               {"SpikeTime": -1.0, "SpikePolarity": 1, "Spiked": False})
        else:
            return NeuronModel(genn_model, None, {}, 
                               {"SpikeTime": -1.0, "Spiked": False})
            
