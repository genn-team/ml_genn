import math
import numpy as np

from pygenn import VarAccess
from ..callbacks import Callback
from ..utils.quantisation import quantise_signed

class SignedWeightQuantise(Callback):
    def __init__(self, synapse_group, weight_var_name,
                 quant_weight_var_name, percentile, num_weight_bits):
        self.synapse_group = synapse_group
        self.weight_var = synapse_group.vars[weight_var_name]
        self.quant_weight_var = synapse_group.vars[quant_weight_var_name]
        self.percentile = percentile
        self.num_weight_bits = num_weight_bits

    def on_batch_begin(self, batch):
        # Download weights
        self.weight_var.pull_from_device()
        
        # Quantise and push to device
        self.quant_weight_var.view[:] = quantise_signed(self.weight_var.view,
                                                        self.num_weight_bits,
                                                        self.percentile)
        self.quant_weight_var.push_to_device()