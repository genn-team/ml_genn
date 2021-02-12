
class BaseNeurons(object):

    def __init__(self):
        self.signed_spikes = False
        self.nrn = None

    def compile(self, mlg_model, name, n, model, params, vars, egp):
        self.nrn = mlg_model.g_model.add_neuron_population(
            name, n, model, params, vars)
        for p in egp:
            self.nrn.set_extra_global_param(p, egp[p])
