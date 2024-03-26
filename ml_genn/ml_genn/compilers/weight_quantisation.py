import logging
import math
import numpy as np

from ..callbacks import Callback
from ..utils.quantisation import quantise_signed

logger = logging.getLogger(__name__)

class WeightQuantiseBase(Callback):
    def __init__(self, synapse_groups, weight_var_name,
                 quant_weight_var_name, percentile, num_weight_bits):
        self.synapse_groups = synapse_groups
        self.weight_var_name = weight_var_name
        self.quant_weight_var_name = quant_weight_var_name
        self.percentile = percentile
        self.num_weight_bits = num_weight_bits

    def _apply(self):
        # Download weights
        for s in self.synapse_groups:
            s.pull_var_from_device(self.weight_var_name)

        # Build stack of weight views
        weight_views = np.stack([s.vars[self.weight_var_name].view
                                 for s in self.synapse_groups])

        # Quantise together
        quant_weights = quantise_signed(weight_views, self.num_weight_bits,
                                        self.percentile)

        # Loop through synapse groups
        for i, s in enumerate(self.synapse_groups):
            # Copy row of quantised weights into view
            quant_view = s.vars[self.quant_weight_var_name].view
            quant_view[:] = quant_weights[i,:]

            # Push back to device
            s.push_var_to_device(self.quant_weight_var_name)
        
class WeightQuantiseBatch(WeightQuantiseBase):
    def on_batch_begin(self, batch):
        self._apply()

class WeightQuantiseTrain(WeightQuantiseBase):
    def on_train_begin(self):
        self._apply()

class WeightQuantiseTest(WeightQuantiseBase):
    def on_test_begin(self):
        self._apply()