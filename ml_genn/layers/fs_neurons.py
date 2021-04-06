import numpy as np
from pygenn.genn_model import create_dpf_class, create_custom_neuron_class
from ml_genn.layers.neurons import Neurons

fs_relu_first_phase_model = create_custom_neuron_class(
    'fs_relu_first_phase',
    param_names=['K', 'alpha'],
    derived_params=[("scale", create_dpf_class(lambda pars, dt: pars[1] * 2**(-pars[0]))())],
    var_name_types=[('Fx', 'scalar'), ('Vmem', 'scalar')],
    sim_code='''
    // Convert K to integer
    const int kInt = (int)$(K);

    // Get timestep within presentation
    const int pipeTimestep = (int)($(t) / DT) % (2 * kInt);

    // Calculate magic constant which, for RelU C=T=h=d
    const scalar C = $(scale) * (1 << (kInt - (1 + pipeTimestep)));

    // If this is the first timestep, apply input
    if(pipeTimestep == 0) {
        $(Vmem) = $(Fx);
        $(Fx) = 0.0;
    }

    // Accumulate input
    $(Fx) += ($(Isyn) * C);
    ''',
    threshold_condition_code='''
    pipeTimestep < kInt && $(Vmem) > C
    ''',
    reset_code='''
    $(Vmem) -= C;
    ''',
    is_auto_refractory_required=False)

fs_relu_second_phase_model = create_custom_neuron_class(
    'fs_relu_second_phase',
    param_names=['K', 'alpha'],
    derived_params=[("scale", create_dpf_class(lambda pars, dt: pars[1] * 2**(-pars[0]))())],
    var_name_types=[('Fx', 'scalar'), ('Vmem', 'scalar')],
    sim_code='''
    // Convert K to integer
    const int kInt = (int)$(K);

    // Get timestep within presentation
    const int pipeTimestep = (int)($(t) / DT) % (2 * kInt);

    // Calculate magic constant which, for RelU C=T=h=d
    const scalar C = $(scale) * (1 << (kInt - (1 + pipeTimestep)));

    // If this is the first timestep, apply input
    if(pipeTimestep == kInt) {
        $(Vmem) = $(Fx);
        $(Fx) = 0.0;
    }

    // Accumulate input
    $(Fx) += ($(Isyn) * C);
    ''',
    threshold_condition_code='''
    pipeTimestep >= kInt && $(Vmem) > C
    ''',
    reset_code='''
    $(Vmem) -= C;
    ''',
    is_auto_refractory_required=False)
    
class FSReluNeurons(Neurons):
    def __init__(self, first_phase, K=10, alpha=25):
        super(FSReluNeurons, self).__init__()
        self.first_phase = first_phase
        self.K = K
        self.alpha = alpha
    
    def compile(self, mlg_model, name, n):
        model = (fs_relu_first_phase_model if self.first_phase
                 else fs_relu_second_phase_model)
        params = {'K': self.K, 'alpha': self.alpha}
        vars = {'Fx': 0.0, 'Vmem': 0}

        super(FSReluNeurons, self).compile(mlg_model, name, n, model,
                                           params, vars, {})

    def set_threshold(self, threshold):
        raise NotImplementedError('Few Spike neurons do not have '
                                  'overridable thresholds')

    def get_predictions(self):
        self.nrn.pull_var_from_device('Fx')
        if self.nrn.vars['Fx'].view.ndim == 1:
            output_view = self.nrn.vars['Fx'].view[np.newaxis]
        else:
            output_view = self.nrn.vars['Fx'].view[:batch_n]

        return output_view.argmax(axis=1)
