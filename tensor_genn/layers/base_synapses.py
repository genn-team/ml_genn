from weakref import proxy

class BaseSynapses(object):

    def __init__(self):
        self.name = None
        self.source = None
        self.target = None
        self.weights = None
        self.syn = None

    def connect(self, source, target):
        self.name = '{}_to_{}_syn'.format(source.name, target.name)
        self.source = proxy(source)
        self.target = proxy(target)
        source.downstream_synapses.append(self)
        target.upstream_synapses.append(self)

    def set_weights(self, weights):
        self.weights[:] = weights

    def get_weights(self):
        return self.weights.copy()

    def compile(self, tg_model, conn, delay, wu_model, wu_param, wu_var,
                wu_pre_var, wu_post_var, ps_model, ps_param, ps_var, conn_init):
        self.syn = [None] * tg_model.batch_size

        # Add batch synapse populations
        for i, (pre, post) in enumerate(zip(self.source.neurons.nrn, self.target.neurons.nrn)):
            name = '{}_{}'.format(self.name, i)

            # Batch master
            if not tg_model.share_weights or i == 0:
                self.syn[i] = tg_model.g_model.add_synapse_population(
                    name, conn, delay, pre, post,
                    wu_model, wu_param, wu_var, wu_pre_var, wu_post_var,
                    ps_model, ps_param, ps_var,
                    conn_init)

            # Batch slave
            else:
                master_name = '{}_0'.format(self.name)
                self.syn[i] = tg_model.g_model.add_slave_synapse_population(
                    name, master_name, delay, pre, post, 'DeltaCurr', {}, {})
