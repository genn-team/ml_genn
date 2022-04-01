from .few_spike_relu_input import FewSpikeReluInput
from .neuron import Neuron
from ..utils import InitValue, NeuronModel, Value

# Standard FS ReLU model where upstream neurons are FS ReLU or FS unsigned input
genn_model = {
    "param_name_types": [("K", "int"), ("Scale", "scalar"), 
                         ("SourceScale", "scalar")],
    "var_name_types": [("Fx", "scalar"), ("V", "scalar")],

    "sim_code": 
        """
        // Convert K to integer
        const int kInt = (int)$(K);

        // Get timestep within presentation
        const int pipeTimestep = (int)($(t) / DT);

        // Calculate magic constants. For RelU hT=h=T
        // **NOTE** d uses last timestep as that was when spike was SENT
        const scalar hT = $(Scale) * (1 << (kInt - (1 + pipeTimestep)));
        const scalar d = $(SourceScale) * (1 << ((kInt - pipeTimestep) % kInt));

        // Accumulate input
        // **NOTE** needs to be before applying input as spikes from LAST timestep must be processed
        $(Fx) += ($(Isyn) * d);
        """,
    "threshold_condition_code":
        """
        $(V) >= hT
        """,
    "reset_code":
        """
        $(V) -= hT;
        """,
    "is_auto_refractory_required": False}

# FS ReLU model where upstream neurons are FS signed input
genn_model_upstream_signed = {
    "param_name_types": [("K", "int"), ("Scale", "scalar"), 
                         ("SourceScale", "scalar")],
    "var_name_types": [("Fx", "scalar"), ("V", "scalar")],
    "sim_code":
        """
        // Convert K to integer
        const int kInt = (int)$(K);

        // Get timestep within presentation
        const int pipeTimestep = (int)($(t) / DT);

        // Calculate magic constants. For RelU hT=h=T
        const scalar hT = $(Scale) * (1 << (kInt - (1 + pipeTimestep)));
        
        // Split timestep into interleaved positive and negative
        // **NOTE** sign is flipped compared to input model as we want sign of PREVIOUS timestep
        const scalar dSign = ((pipeTimestep % 2) == 0) ? -1.0 : 1.0;
        const scalar d = dSign * $(SourceScale) * (1 << (((kInt - pipeTimestep) % kInt) / 2));
        
        // Accumulate input
        // **NOTE** needs to be before applying input as spikes from LAST timestep must be processed
        $(Fx) += ($(Isyn) * d);
        """,
    "threshold_condition_code":
        """
        $(V) >= hT
        """,
    "reset_code":
        """
        $(V) -= hT;
        """,
    "is_auto_refractory_required": False}

class FewSpikeRelu(Neuron):
    pipelined = True

    def __init__(self, k=10, alpha=25, output=None):
        super(FewSpikeRelu, self).__init__(output)
        self.k = Value(k)
        self.alpha = Value(alpha)
        
        if not self.k.is_constant or not self.alpha.is_constant:
            raise NotImplementedError("FewSpike ReLU model currently requires"
                                      " homogeneous k and alpha values")
    
    def get_model(self, population, dt):
         # Loop through incoming connections
        source_alpha = None
        source_signed = None
        for c in population.incoming_connections:
            # Get neuron object associated with the source layer
            nrn = c().source().neuron
            
            # If the upstream neuron is some sort of FsRelu
            # **YUCK** is there a better way of accessing the FsReluNeurons type?
            source_relu = isinstance(nrn, type(self))
            source_relu_input = isinstance(nrn, FewSpikeReluInput)
            if source_relu or source_relu_input:
                # Check K parameters match
                if nrn.k.value != self.k.value:
                    raise ValueError("K parameters of FewSpike ReLU neurons "
                                     "must match across whole model")
                
                # Check that all upstream neurons have the same alpha 
                if source_alpha is None:
                    source_alpha = nrn.alpha
                elif source_alpha.value != nrn.alpha.value:
                    raise ValueError("All upstream FewSpike ReLU neurons "
                                     "must have the same alpha values")

                # Check that all upstream neurons match signedness
                nrn_signed = nrn.signed_input if source_relu_input else False
                if source_signed is None:
                    source_signed = nrn_signed
                elif source_signed != nrn_signed:
                    raise ValueError("All upstream FewSpike ReLU input "
                                     "neurons must have the same signedness")
                    
            # Otherwise, give error
            else:
                raise ValueError("FewSpike neurons can only be connected "
                                 "to other FewSpike neurons") 

        # If no source population is found, use our own alpha
        # **NOTE** this shouldn't be necessary
        if source_alpha is None:
            source_alpha = self.alpha

        # Calculate scale
        if source_signed:
            source_scale = source_alpha.value * 2**(-self.k.value // 2)
        else:
            source_scale = source_alpha.value * 2**(-self.k.value)

        scale =  self.alpha.value * 2**(-self.k.value)

        model = genn_model_upstream_signed if source_signed else genn_model
        return self.add_output_logic(
            NeuronModel(model, 
                        {"K": self.k, "Scale": Value(scale), 
                         "SourceScale": Value(source_scale)},
                        {"Fx": Value(0.0), "V": Value(0.0)}), "Fx")

