import numpy as np

from . import Connectivity

class Dense(Connectivity):
    def __init__(self):
        super(Connectivity, self).__init__()

    def connect(self, source_pop, target_pop):
        super(Dense, self).connect(source_pop, target_pop)

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
