from pygenn.genn_model import create_custom_neuron_class
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from ml_genn.layers.input_neurons import InputNeurons

poisson_input_model = create_custom_neuron_class(
    'poisson_input',
    var_name_types=[('input', 'scalar', VarAccess_READ_ONLY)],
    sim_code='''
    const bool spike = $(gennrand_uniform) >= exp(-fabs($(input)) * DT);
    ''',
    threshold_condition_code='''
    $(input) > 0.0 && spike
    ''',
    is_auto_refractory_required=False,
)

class PoissonInputNeurons(InputNeurons):

    def __init__(self, signed_spikes=False):
        model = poisson_input_model
        params = {}
        vars_init = {'input': 0.0}
        global_params = {}
        super(PoissonInputNeurons, self).__init__(model, params, vars_init, global_params)
        self.signed_spikes = signed_spikes
