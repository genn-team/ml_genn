import math
import numpy as np

from ..callbacks import Callback
from ..utils.quantisation import find_signed_scale

class SignedWeightQuantise(Callback):
    def __init__(self, synapse_group, weight_var_name,
                 percentile, num_weight_bits):
        self.synapse_group = synapse_group
        self.weight_var = synapse_group.vars[weight_var_name]
        self.percentile = percentile
        self.num_weight_bits = num_weight_bits

    def on_batch_begin(self, batch):
        # Download weights
        self.weight_var.pull_from_device()
        
        # Find scaling factors
        min_quant, max_quant, scale = find_signed_scale(self.weight_var.view,
                                                        self.num_weight_bits,
                                                        self.percentile)

        # Apply to synapse group
        self.synapse_group.set_dynamic_param_value("MinWeight", min_quant)
        self.synapse_group.set_dynamic_param_value("MaxWeight", max_quant)
        self.synapse_group.set_dynamic_param_value("WeightScale", scale)

def add_wum_quantisation(wum, weight_var_name):
    # Add quantization parameters to weight update model
    wum.add_param("MinWeight", "scalar", 0.0)
    wum.add_param("MaxWeight", "scalar", 0.0)
    wum.add_param("WeightScale", "scalar", 0.0)

    # Make all dynamic
    wum.set_param_dynamic("MinWeight")
    wum.set_param_dynamic("MaxWeight")
    wum.set_param_dynamic("WeightScale")

    # Add code to inject quantised
    wum.append_pre_spike_syn_code(
        f"addToPost(fmin(MaxWeight, fmax(MinWeight, WeightScale * round({weight_var_name} / WeightScale))));")