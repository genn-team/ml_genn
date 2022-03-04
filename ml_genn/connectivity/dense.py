import numpy as np

from ml_genn.layers.base_synapses import BaseSynapses
from ml_genn.layers.weight_update_models import signed_static_pulse

class DenseSynapses(BaseSynapses):

    def __init__(self, units):
        super(DenseSynapses, self).__init__()
        self.units = units

    def connect(self, source, target):
        super(DenseSynapses, self).connect(source, target)

        output_shape = (self.units, )

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((np.prod(source.shape), self.units), dtype=np.float64)

    def compile(self, mlg_model, name):
        conn = 'DENSE_INDIVIDUALG'
        wu_model = signed_static_pulse if self.source().neurons.signed_spikes else 'StaticPulse'
        wu_var = {'g': self.weights.flatten()}

        super(DenseSynapses, self).compile(mlg_model, name, conn, wu_model, {}, wu_var,
                                           {}, {}, 'DeltaCurr', {}, {}, None, {})
