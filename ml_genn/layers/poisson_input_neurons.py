from pygenn.genn_model import create_custom_neuron_class
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from ml_genn.layers.input_neurons import InputNeurons

poisson_input_model = create_custom_neuron_class(
    'poisson_input',
    var_name_types=[('input', 'scalar', VarAccess_READ_ONLY_DUPLICATE)],
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
        super(PoissonInputNeurons, self).__init__()
        self.signed_spikes = signed_spikes

    def compile(self, mlg_model, layer):
        model = poisson_input_model
        vars = {'input': 0.0}

        super(PoissonInputNeurons, self).compile(mlg_model, layer, 
                                                 model, {}, vars, {})
