from pygenn.genn_model import create_custom_neuron_class
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from tensor_genn.layers.input_neurons import InputNeurons

spike_input_model = create_custom_neuron_class(
    'spike_input',
    var_name_types=[('input', 'scalar', VarAccess_READ_ONLY)],
    sim_code='''
    int spike = input != 0.0;
    ''',
    threshold_condition_code='''
    $(input) > 0.0 && spike
    ''',
    is_auto_refractory_required=False,
)

class SpikeInputNeurons(InputNeurons):

    def __init__(self, signed_spikes=False):
        model = spike_input_model
        params = {}
        vars_init = {'input': 0.0}
        global_params = {}
        super(SpikeInputNeurons, self).__init__(model, params, vars_init, global_params)
        self.signed_spikes = signed_spikes
