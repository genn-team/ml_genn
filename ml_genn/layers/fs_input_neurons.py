from pygenn.genn_model import create_dpf_class, create_custom_neuron_class
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from ml_genn.layers.input_neurons import InputNeurons

fs_relu_input_model = create_custom_neuron_class(
    'fs_relu_input',
    param_names=['K', 'alpha'],
    derived_params=[("scale", create_dpf_class(lambda pars, dt: pars[1] * 2**(-pars[0]))())],
    var_name_types=[('input', 'scalar', VarAccess_READ_ONLY_DUPLICATE), ('Vmem', 'scalar')],
    sim_code='''
    // Convert K to integer
    const int kInt = (int)$(K);
    
    // Get timestep within presentation
    const int pipeTimestep = (int)($(t) / DT);

    // If this is the first timestep, apply input
    if(pipeTimestep == 0) {
        $(Vmem) = $(input);
    }
    
    const scalar hT = $(scale) * (1 << (kInt - (1 + pipeTimestep)));
    ''',
    threshold_condition_code='''
    pipeTimestep < kInt && $(Vmem) >= hT
    ''',
    reset_code='''
    $(Vmem) -= hT;
    ''',
    is_auto_refractory_required=False)

class FSReluInputNeurons(InputNeurons):
    def __init__(self, K=10, alpha=25):
        super(FSReluInputNeurons, self).__init__()
        self.K = K
        self.alpha = alpha

    def compile(self, mlg_model, layer):
        model = fs_relu_input_model
        params = {'K' : self.K, 'alpha': self.alpha}
        vars = {'input': 0.0, 'Vmem': 0.0}

        super(FSReluInputNeurons, self).compile(mlg_model, layer, model,
                                                params, vars, {})
