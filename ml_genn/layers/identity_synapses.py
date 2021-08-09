from pygenn.genn_model import create_custom_sparse_connect_init_snippet_class

from ml_genn.layers.base_synapses import BaseSynapses

identity_init = create_custom_sparse_connect_init_snippet_class(
    'identity',

    row_build_code='''
    $(addSynapse, $(id_pre));
    $(endRow);
    ''',
)

class IdentitySynapses(BaseSynapses):

    def __init__(self):
        super(IdentitySynapses, self).__init__()

    def connect(self, source, target):
        super(IdentitySynapses, self).connect(source, target)

        output_shape = source.shape

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = None

    def compile(self, mlg_model, name):

        conn = ('DENSE_PROCEDURALG' if self.connectivity_type == ConnectivityType.PROCEDURAL
                else 'DENSE_INDIVIDUALG')
        wu_model = signed_static_pulse if self.source().neurons.signed_spikes else 'StaticPulse'
        wu_var = {'g': wu_var_init}

        super(IdentitySynapses, self).compile(mlg_model, name, conn, 0, wu_model, {}, wu_var,
                                              {}, {}, 'DeltaCurr', {}, {}, None, wu_var_egp)
