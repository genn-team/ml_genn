import numpy as np
from pygenn.genn_model import create_dpf_class, create_custom_neuron_class
from ml_genn.layers.fs_input_neurons import FSReluInputNeurons
from ml_genn.layers.neurons import Neurons

# Standard FS ReLU model where upstream neurons are FS ReLU or FS unsigned input
fs_relu_model = create_custom_neuron_class(
    'fs_relu',
    param_names=['K', 'alpha', 'upstreamAlpha'],
    derived_params=[("scale", create_dpf_class(lambda pars, dt: pars[1] * 2**(-pars[0]))()),
                    ("upstreamScale", create_dpf_class(lambda pars, dt: pars[2] * 2**(-pars[0]))())],
    var_name_types=[('Fx', 'scalar'), ('Vmem', 'scalar')],
    sim_code='''
    // Convert K to integer
    const int kInt = (int)$(K);

    // Get timestep within presentation
    const int pipeTimestep = (int)($(t) / DT);

    // Calculate magic constants. For RelU hT=h=T
    // **NOTE** d uses last timestep as that was when spike was SENT
    const scalar hT = $(scale) * (1 << (kInt - (1 + pipeTimestep)));
    const scalar d = $(upstreamScale) * (1 << ((kInt - pipeTimestep) % kInt));

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

# FS ReLU model where upstream neurons are FS signed input
fs_relu_upstream_signed_input_model = create_custom_neuron_class(
    'fs_relu_upstream_signed_input',
    param_names=['K', 'alpha', 'upstreamAlpha'],
    derived_params=[("scale", create_dpf_class(lambda pars, dt: pars[1] * 2**(-pars[0]))()),
                    ("upstreamScale", create_dpf_class(lambda pars, dt: pars[2] * 2**(-pars[0]//2))())],
    var_name_types=[('Fx', 'scalar'), ('Vmem', 'scalar')],
    sim_code='''
    // Convert K to integer
    const int kInt = (int)$(K);

    // Get timestep within presentation
    const int pipeTimestep = (int)($(t) / DT);

    // Calculate magic constants. For RelU hT=h=T
    const scalar hT = $(scale) * (1 << (kInt - (1 + pipeTimestep)));
    
    // Split timestep into interleaved positive and negative
    // **NOTE** sign is flipped compared to input model as we want sign of PREVIOUS timestep
    const scalar dSign = ((pipeTimestep % 2) == 0) ? -1.0 : 1.0;
    const scalar d = dSign * $(upstreamScale) * (1 << (((kInt - pipeTimestep) % kInt) / 2));
    
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
        upstream_alpha = None
        upstream_signed = None
        for u in layer.upstream_synapses:
            # Get neuron object associated with the source layer
            nrn = u.source().neurons
            
            # If the upstream neuron is some sort of FsRelu
            # **YUCK** is there a better way of accessing the FsReluNeurons type?
            upstream_relu = isinstance(nrn, type(self))
            upstream_relu_input = isinstance(nrn, FSReluInputNeurons)
            if upstream_relu or upstream_relu_input:
                # Check K parameters match
                if nrn.K != self.K:
                    raise ValueError("K parameters of FS ReLU neurons must "
                                     "match across whole model")
                
                # Check that all upstream neurons have the same alpha 
                if upstream_alpha is None:
                    upstream_alpha = nrn.alpha
                elif upstream_alpha != nrn.alpha:
                    raise ValueError("All upstream FS ReLU neurons must "
                                     "have the same alpha parameter values")

                # Check that all upstream neurons match signedness
                nrn_signed = nrn.signed_input if upstream_relu_input else False
                if upstream_signed is None:
                    upstream_signed = nrn_signed
                elif upstream_signed != nrn_signed:
                    raise ValueError("All upstream FS ReLU input neurons "
                                     "must  have the same signedness")
                    
            # Otherwise, give error
            else:
                raise ValueError("FS neurons can only be connected "
                                 "to other FS neurons") 

        # If no upstream population is found, use our own alpha
        # **NOTE** this shouldn't be necessary
        if upstream_alpha is None:
            upstream_alpha = self.alpha

        # Pick model based on whether upstream neurons are signed or not
        model = (fs_relu_upstream_signed_input_model if upstream_signed == True
                 else fs_relu_model)

        params = {'K': self.K, 'alpha': self.alpha, 
                  'upstreamAlpha': upstream_alpha}
        vars = {'Fx': 0.0, 'Vmem': 0}

        super(FSReluNeurons, self).compile(mlg_model, layer, model,
                                           params, vars, {})

    def set_threshold(self, threshold):
        raise NotImplementedError('FS neurons do not have '
                                  'overridable thresholds')

    def get_predictions(self, batch_n):
        self.nrn.pull_var_from_device('Fx')
        if self.nrn.vars['Fx'].view.ndim == 1:
            output_view = self.nrn.vars['Fx'].view[np.newaxis]
        else:
            output_view = self.nrn.vars['Fx'].view[:batch_n]
        return output_view.argmax(axis=1)
