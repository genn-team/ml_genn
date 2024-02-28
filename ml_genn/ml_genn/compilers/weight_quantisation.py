import math
import numpy as np

from ..callbacks import Callback
from ..utils.quantisation import quantise_signed

class WeightQuantiseBase(Callback):
    def __init__(self, synapse_group, weight_var_name,
                 quant_weight_var_name, percentile, num_weight_bits):
        self.synapse_group = synapse_group
        self.weight_var_name = weight_var_name
        self.quant_weight_var_name = quant_weight_var_name
        self.percentile = percentile
        self.num_weight_bits = num_weight_bits

    def _apply(self):
        # Download weights
        self.synapse_group.pull_var_from_device(self.weight_var_name)
        
        # Quantise and push to device
        view = self.synapse_group.vars[self.weight_var_name].view
        quant_view = self.synapse_group.vars[self.quant_weight_var_name].view
        quant_view[:] = quantise_signed(view, self.num_weight_bits,
                                        self.percentile)
        self.synapse_group.push_var_to_device(self.quant_weight_var_name)
        
class WeightQuantiseBatch(WeightQuantiseBase):
    def on_batch_begin(self, batch):
        self._apply()

class WeightQuantiseTrain(WeightQuantiseBase):
    def on_train_begin(self):
        self._apply()