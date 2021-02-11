from weakref import proxy

class BaseSynapses(object):

    def __init__(self):
        self.source = None
        self.target = None
        self.weights = None
        self.syn = None

    def connect(self, source, target):
        self.source = proxy(source)
        self.target = proxy(target)
        source.downstream_synapses.append(self)
        target.upstream_synapses.append(self)

    def set_weights(self, weights):
        self.weights[:] = weights

    def get_weights(self):
        return self.weights.copy()

    def compile(self, mlg_model, name, conn, delay,
                wu_model, wu_params, wu_vars,
                wu_pre_vars, wu_post_vars,
                ps_model, ps_params, ps_vars,
                conn_init, wu_vars_egp):
        self.syn = mlg_model.g_model.add_synapse_population(
            name, conn, delay, self.source.neurons.nrn, self.target.neurons.nrn,
            wu_model, wu_params, wu_vars, wu_pre_vars, wu_post_vars,
            ps_model, ps_params, ps_vars, conn_init)
        for wu_var, wu_var_egp in zip(wu_vars_egp.keys(), wu_vars_egp.values()):
            for p, value in zip(wu_var_egp.keys(), wu_var_egp.values()):
                self.syn.vars[wu_var].set_extra_global_init_param(p, value)
