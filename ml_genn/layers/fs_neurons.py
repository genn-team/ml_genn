import numpy as np
from pygenn.genn_model import create_dpf_class, create_custom_neuron_class
from ml_genn.layers.fs_input_neurons import FSReluInputNeurons
from ml_genn.layers.neurons import Neurons

fs_relu_model = create_custom_neuron_class(
    'fs_relu',
    param_names=['K', 'alpha'],
    derived_params=[("scale", create_dpf_class(lambda pars, dt: pars[1] * 2**(-pars[0]))())],
    var_name_types=[('Fx', 'scalar'), ('Vmem', 'scalar')],
    sim_code='''
    // Convert K to integer
    const int kInt = (int)$(K);

    // Get timestep within presentation
    const int pipeTimestep = (int)($(t) / DT);

    // Calculate magic constants. For RelU hT=h=T
    // **NOTE** d uses last timestep as that was when spike was SENT
    const scalar hT = $(scale) * (1 << (kInt - (1 + pipeTimestep)));
    const scalar d = $(scale) * (1 << ((kInt - pipeTimestep) % kInt));

    // Accumulate input
    // **NOTE** needs to be before applying input as spikes from LAST timestep must be processed
    $(Fx) += ($(Isyn) * d);

    // If this is the first timestep, apply input
    if(pipeTimestep == 0) {
        $(Vmem) = $(Fx);
        $(Fx) = 0.0;
    }
    ''',
    threshold_condition_code='''
    $(Vmem) >= hT
    ''',
    reset_code='''
    $(Vmem) -= hT;
    ''',
    is_auto_refractory_required=False)

class FSReluNeurons(Neurons):
    pipelined = True

    def __init__(self, K=10, alpha=25):
        super(FSReluNeurons, self).__init__()
        self.K = K
        self.alpha = alpha

    def compile(self, mlg_model, layer):
        # Loop through upstream synapses
        print(layer.name)
        for u in layer.upstream_synapses:
            # Get neuron object associated with the source layer
            nrn = u.source.neurons
            
            # If the upstream neuron is another FsRelu
            # **YUCK** is there a better way of accessing the FsReluNeurons type?
            if isinstance(nrn, type(self)):
                # Check K parameters match
                if nrn.K != self.K:
                    raise ValueError("K parameters of FS neurons must "
                                     "match across whole model")
             
                print("\tSource is FsRelu")
            # Otherwise, if it's a FSReluInput
            elif isinstance(nrn, FSReluInputNeurons):
                # Check K parameters match
                if nrn.K != self.K:
                    raise ValueError("K parameters of FS neurons must "
                                     "match across whole model")
                
                print("\tSource is FsReluInput")
            # Otherwise, give error
            else:
                raise ValueError("Few spike neurons can only be "
                                 "connected to other few spike neurons") 

        params = {'K': self.K, 'alpha': self.alpha}
        vars = {'Fx': 0.0, 'Vmem': 0}

        super(FSReluNeurons, self).compile(mlg_model, layer, fs_relu_model,
                                           params, vars, {})

    def set_threshold(self, threshold):
        raise NotImplementedError('Few Spike neurons do not have '
                                  'overridable thresholds')

    def get_predictions(self, batch_n):
        self.nrn.pull_var_from_device('Fx')
        if self.nrn.vars['Fx'].view.ndim == 1:
            output_view = self.nrn.vars['Fx'].view[np.newaxis]
        else:
            output_view = self.nrn.vars['Fx'].view[:batch_n]
        return output_view.argmax(axis=1)
