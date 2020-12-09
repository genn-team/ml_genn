from pygenn.genn_model import create_custom_neuron_class
from tensor_genn.layers.neurons import Neurons

if_model = create_custom_neuron_class(
    'if',
    var_name_types=[('Vmem', 'scalar'), ('nSpk', 'unsigned int')],
    extra_global_params=[('Vthr', 'scalar')],
    sim_code='''
    if ($(t) == 0.0) {
        // Reset state at t = 0
        $(Isyn) = 0.0;
        $(Vmem) = 0.0;
        $(nSpk) = 0;
    }
    $(Vmem) += $(Isyn) * DT;
    ''',
    threshold_condition_code='''
    $(Vmem) >= $(Vthr)
    ''',
    reset_code='''
    $(Vmem) = 0.0;
    $(nSpk) += 1;
    ''',
    is_auto_refractory_required=False,
)

class IFNeurons(Neurons):

    def __init__(self, threshold=1.0):
        model = if_model
        params = {}
        vars_init = {'Vmem': 0.0, 'nSpk': 0}
        global_params = {'Vthr': threshold}
        super(IFNeurons, self).__init__(model, params, vars_init, global_params)

    def set_threshold(self, threshold):
        self.global_params['Vthr'] = threshold

        if self.nrn is not None:
            for n in self.nrn:
                n.extra_global_params['Vthr'].view[:] = threshold
