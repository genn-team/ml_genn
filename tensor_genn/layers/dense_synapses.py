import numpy as np
from pygenn.genn_wrapper import NO_DELAY

from tensor_genn.layers.base_synapses import BaseSynapses
from tensor_genn.layers.weight_update_models import signed_static_pulse

class DenseSynapses(BaseSynapses):

    def __init__(self, units):
        super(DenseSynapses, self).__init__()
        self.units = units

    def connect(self, source, target):
        super(DenseSynapses, self).connect(source, target)

        output_shape = (self.units, )

        if target.neurons.shape is None:
            target.neurons.shape = output_shape
        elif output_shape != target.neurons.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((np.prod(source.neurons.shape), self.units), dtype=np.float64)

    def compile(self, tg_model):

        conn = 'DENSE_INDIVIDUALG'

        wu_model = signed_static_pulse if self.source.neurons.signed_spikes else 'StaticPulse'

        wu_var = {'g': self.weights.flatten()}
        wu_var_egp = {'g': {'weights': self.weights.flatten()}}

        super(DenseSynapses, self).compile(tg_model, conn, 0, wu_model, {}, wu_var, {},
                                           {}, {}, 'DeltaCurr', {}, {}, None)
