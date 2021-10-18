import numpy as np
from pygenn.genn_model import init_connectivity

from ml_genn.layers import ConnectivityType
from ml_genn.layers.base_synapses import BaseSynapses

class IdentitySynapses(BaseSynapses):

    def __init__(self, connectivity_type='procedural'):
        super(IdentitySynapses, self).__init__()
        self.connectivity_type = ConnectivityType(connectivity_type)

    def connect(self, source, target):
        super(IdentitySynapses, self).connect(source, target)

        output_shape = source.shape

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty(0, dtype=np.float64)

    def compile(self, mlg_model, name):
        conn_init = init_connectivity('OneToOne', {})
        conn = ('PROCEDURAL_GLOBALG' if self.connectivity_type == ConnectivityType.PROCEDURAL
                else 'SPARSE_GLOBALG')
        wu_model = signed_static_pulse if self.source().neurons.signed_spikes else 'StaticPulse'
        wu_var = {'g': 1.0}

        super(IdentitySynapses, self).compile(mlg_model, name, conn, wu_model, {}, wu_var,
                                              {}, {}, 'DeltaCurr', {}, {}, conn_init, {})
