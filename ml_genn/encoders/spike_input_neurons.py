from pygenn.genn_model import create_custom_neuron_class
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from ml_genn.layers.input_neurons import InputNeurons

spike_input_model = create_custom_neuron_class(
    'spike_input',
    var_name_types=[('input', 'scalar', VarAccess_READ_ONLY_DUPLICATE)],
    sim_code='''
    const bool spike = $(input) != 0.0;
    ''',
    threshold_condition_code='''
    $(input) > 0.0 && spike
    ''',
    is_auto_refractory_required=False,
)

class SpikeInputNeurons(InputNeurons):

    def __init__(self, signed_spikes=False):
        super(SpikeInputNeurons, self).__init__()
        self.signed_spikes = signed_spikes

    def compile(self, mlg_model, layer):
        model = spike_input_model
        vars = {'input': 0.0}

        super(SpikeInputNeurons, self).compile(mlg_model, layer, 
                                               model, {}, vars, {})
