from . import Connectivity
from ..utils import InitValue, Value

class OneToOne(Connectivity):
    def __init__(self, weight:InitValue, delay:InitValue=0):
        super(OneToOne, self).__init__(weight, delay)

    def connect(self, source, target):
        output_shape = source.shape

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError("Target population shape mismatch")

    def compile(self, mlg_model, name):
        conn_init = init_connectivity('OneToOne', {})
        conn = ('PROCEDURAL_GLOBALG' if self.connectivity_type == ConnectivityType.PROCEDURAL
                else 'SPARSE_GLOBALG')
        wu_model = signed_static_pulse if self.source().neurons.signed_spikes else 'StaticPulse'
        wu_var = {'g': 1.0}

        super(IdentitySynapses, self).compile(mlg_model, name, conn, wu_model, {}, wu_var,
                                              {}, {}, 'DeltaCurr', {}, {}, conn_init, {})
