from pygenn.genn_model import create_custom_neuron_class
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from ml_genn.layers.input_neurons import InputNeurons

if_input_model = create_custom_neuron_class(
    'if_input',
    var_name_types=[('input', 'scalar', VarAccess_READ_ONLY_DUPLICATE), ('Vmem', 'scalar')],
    sim_code='''
    if ($(t) == 0.0) {
        // Reset state at t = 0
        $(Vmem) = 0.0;
    }
    $(Vmem) += $(input) * DT;
    ''',
    threshold_condition_code='''
    $(Vmem) >= 1.0
    ''',
    reset_code='''
    $(Vmem) = 0.0;
    ''',
    is_auto_refractory_required=False,
)

class IFInputNeurons(InputNeurons):

    def compile(self, mlg_model, layer):
        model = if_input_model
        vars = {'input': 0.0, 'Vmem': 0.0}

        super(IFInputNeurons, self).compile(mlg_model, layer, model, {}, vars, {})
