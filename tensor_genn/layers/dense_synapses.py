import numpy as np
from pygenn.genn_wrapper import NO_DELAY

from tensor_genn.layers.base_synapses import BaseSynapses
from tensor_genn.layers.weight_update_models import signed_static_pulse

class DenseSynapses(BaseSynapses):

    def __init__(self, units):
        super(DenseSynapses, self).__init__()
        self.units = units

    def compile(self, tg_model):
        super(DenseSynapses, self).compile(tg_model)

        # Add batch synapse populations
        for batch_i in range(tg_model.batch_size):
            pre_nrn = self.source.nrn[batch_i]
            post_nrn = self.target.nrn[batch_i]
            syn_name = '{}_to_{}_syn_{}'.format(self.source.name, self.target.name, batch_i)

            # Batch master
            if not tg_model.share_weights or batch_i == 0:
                model = signed_static_pulse if self.source.signed_spikes else 'StaticPulse'

                self.syn[batch_i] = tg_model.g_model.add_synapse_population(
                    syn_name, 'DENSE_INDIVIDUALG', NO_DELAY, pre_nrn, post_nrn,
                    model, {}, {'g': self.weights.flatten()}, {}, {}, 'DeltaCurr', {}, {})

            # Batch slave
            else:
                master_syn_name = '{}_to_{}_syn_0'.format(self.source.name, self.target.name)
                self.syn[batch_i] = tg_model.g_model.add_slave_synapse_population(
                    syn_name, master_syn_name, NO_DELAY, pre_nrn, post_nrn, 'DeltaCurr', {}, {})

    def connect(self, source, target):
        super(DenseSynapses, self).connect(source, target)

        self.output_shape = (self.units, )

        if target.shape is None:
            target.shape = self.output_shape
        elif self.output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((np.prod(source.shape), self.units), dtype=np.float64)
