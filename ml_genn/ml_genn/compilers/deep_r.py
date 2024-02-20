import numpy as np

from string import Template
from pygenn import VarAccessMode
from ..callbacks import Callback
from ..utils.model import CustomConnectivityUpdateModel, CustomUpdateModel

from copy import deepcopy
from pygenn import create_egp_ref, create_pre_var_ref

deep_r_l1_model_template = {
    "params": [("C", "scalar")],
    "var_refs": [("Variable", "scalar")],
    "update_code": Template("""
    Variable $assign C;
    """)}
    
deep_r_init_model = {
    "params": [("NumRowWords", "unsigned int")],
    "extra_global_param_refs": [("Connectivity", "uint32_t*")],
    "row_update_code": """
    // Get pointer to start of this row of connectivity
    uint32_t *rowConnectivity = Connectivity + (NumRowWords * id_pre);
    
    // Zero it
    for(int i = 0; i < NumRowWords; i++) {
        rowConnectivity[i] = 0;
    }
    
    // Set connectivity where there are synapse
    for_each_synapse {
        rowConnectivity[id_post / 32] |= (1 << (id_post % 32));
    }
    """}

deep_r_1_model_template = {
    "params": [("NumRowWords", "unsigned int")],
    "var_refs": [("g", "scalar", VarAccessMode.READ_ONLY)],
    "pre_var_refs": [("NumDormant", "unsigned int")],
    "extra_global_param_refs": [("Connectivity", "uint32_t*")],

    "row_update_code": Template("""
    // Zero dormant counter
    NumDormant = 0;

    // Loop through synapses
    uint32_t *rowConnectivity = Connectivity + (NumRowWords * id_pre);
    for_each_synapse {
        // If synapse sign changed
        if(g $comp 0.0) {
            // Increment dormant counter
            NumDormant++;
            
            // Clear connectivity bit
            rowConnectivity[id_post / 32] &= ~(1 << (id_post % 32));

            // Remove synapse
            remove_synapse();
        }
    }
    """)}

deep_r_2_model = {
    "params": [("NumRowWords", "unsigned int")],
    "pre_vars": [("NumDormant", "unsigned int"), ("NumActivations", "unsigned int")],
    "extra_global_params": [("Connectivity", "uint32_t*"), ("NumRewirings", "uint64_t*")],
    
    "host_update_code": """
    pullNumDormantFromDevice();
    pullRowLengthFromDevice();
    
    // Count total dormant and zero activations
    size_t numRemainingDormant = 0;
    size_t numSynapses = 0;
    for(unsigned int i = 0; i < num_pre; i++) {
        numRemainingDormant += NumDormant[i];
        numSynapses += row_length[i];
        NumActivations[i] = 0;
    }

    // Record number of rewirings
    NumRewirings[0] = numRemainingDormant;

    // Loop through all rows but last
    size_t numTotalPaddingSynapses = (row_stride * num_pre) - numSynapses;
    for(unsigned int i = 0; i < (num_pre - 1); i++) {
        const unsigned int numRowPaddingSynapses = row_stride - row_length[i];
        if(numRowPaddingSynapses > 0 && numTotalPaddingSynapses > 0) {
            const scalar prob = (scalar)numRowPaddingSynapses / numTotalPaddingSynapses;
            
            const unsigned int numRowActivations = min(numRowPaddingSynapses,
                                                       gennrand_binomial(numRemainingDormant, prob));
            NumActivations[i] = numRowActivations;
            
            numRemainingDormant -= numRowActivations;
            numTotalPaddingSynapses -= numRowPaddingSynapses;
        }
    }
    
    // Put remaining dormant synapses in last row
    NumActivations[num_pre - 1] = numRemainingDormant;
    
    pushNumActivationsToDevice();
    """,
    
    "row_update_code": """
    // Loop through synapses to activate
    uint32_t *rowConnectivity = Connectivity + (NumRowWords * id_pre);
    for(unsigned int i = 0; i < NumActivations; i++) {
        while(true) {
            // Pick a random synapse to activate
            const unsigned int j = gennrand() % num_post;
            
            if(!(rowConnectivity[j / 32] & (1 << (j % 32)))) {
                add_synapse(j);
                rowConnectivity[j / 32] |= (1 << (j % 32));
                break;
            }
        }
    }
    """}

class RewiringRecord(Callback):
    def __init__(self, deep_r_2_ccu, key=None):
        self.num_rewirings = deep_r_2_ccu.extra_global_params["NumRewirings"]
        self.key = key

    def set_params(self, data, **kwargs):
        # Create empty list to hold recorded data
        data[self.key] = []
        self._data = data[self.key]

    def on_batch_end(self, batch, metrics):
        # Read number of rewirings out of view and add to list
        self._data.append(self.num_rewirings.view[0])
    
    def get_data(self):
        return self.key, self._data


def add_deep_r(synapse_group, genn_model, compiler, l1_strength, 
               delta_weight_var_ref, weight_var_ref, excitatory):
    # Calculate bitmask sizes 
    num_row_words = (synapse_group.trg.num_neurons + 31) // 32
    num_words = synapse_group.src.num_neurons * num_row_words

    # If L1 regularization is enabled
    if l1_strength > 0.0:
        # Make copy of L1 model with correct assignment operator
        # **NOTE** regularization is applied to GRADIENTS which are
        # SUBTRACTED from weight by optimizer so we want to subtract
        # positive values to push excitatory weights towards zero and
        # negative values to push inhibitory weights towards zero
        assign = "+=" if excitatory else "-="
        deep_r_l1_model = deepcopy(deep_r_l1_model_template)
        deep_r_l1_model["update_code"] =\
            deep_r_l1_model["update_code"].substitute(assign=assign)

        # Create custom update model to apply L1 regularisation in correct direction
        deep_r_l1 = CustomUpdateModel(
            deep_r_l1_model,
            param_vals={"C": l1_strength},
            var_refs={"Variable": delta_weight_var_ref})

        compiler.add_custom_update(
            genn_model, deep_r_l1,
            "DeepRL1", "DeepRL1" + synapse_group.name)

    # Make copy of first pass model with correct comparison operator
    comp = "<" if excitatory else ">"
    deep_r_1_model = deepcopy(deep_r_1_model_template)
    deep_r_1_model["row_update_code"] =\
        deep_r_1_model["row_update_code"].substitute(comp=comp)

    # Create custom connectivity update model to
    # implement second deep-r pass and add to model
    # **NOTE** create this first as this has CPU update 
    # which, currently, can't access variable references
    deep_r_2 = CustomConnectivityUpdateModel(
        deep_r_2_model, param_vals={"NumRowWords": num_row_words},
        pre_var_vals={"NumDormant": 0, "NumActivations": 0},
        egp_vals={"Connectivity": np.zeros(num_words, dtype=np.uint32),
                  "NumRewirings": np.zeros(1, dtype=np.uint64)})

    genn_deep_r_2 = compiler.add_custom_connectivity_update(
        genn_model, deep_r_2, synapse_group,
        "DeepR2", "DeepR2" + synapse_group.name)

    # Create custom connectivity update model to  
    # implement first deep-r pass and add to model
    deep_r_1 = CustomConnectivityUpdateModel(
        deep_r_1_model, param_vals={"NumRowWords": num_row_words}, 
        pre_var_refs={"NumDormant": create_pre_var_ref(genn_deep_r_2, "NumDormant")},
        var_refs={"g": weight_var_ref},
        egp_refs={"Connectivity": create_egp_ref(genn_deep_r_2, "Connectivity")})

    compiler.add_custom_connectivity_update(
        genn_model, deep_r_1, synapse_group, 
        "DeepR1", "DeepR1" + synapse_group.name)

    # Create custom connectivity update model to  
    # implement deep-r initialisation and add to model
    deep_r_init = CustomConnectivityUpdateModel(
        deep_r_init_model, param_vals={"NumRowWords": num_row_words}, 
        egp_refs={"Connectivity": create_egp_ref(genn_deep_r_2, "Connectivity")})

    compiler.add_custom_connectivity_update(
        genn_model, deep_r_init, synapse_group, 
        "DeepRInit", "DeepRInit" + synapse_group.name)

    # Return GeNN custom connectivity update used for second pass
    return genn_deep_r_2

