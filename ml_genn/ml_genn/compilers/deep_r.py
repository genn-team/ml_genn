
deep_r_build_connectivity_model = {
    "params": [("RowWords", "unsigned int")],
    "extra_global_param_refs": [("Connectivity", "uint32_t*"), ("SignChange", "uint32_t*"],
    "row_update_code": """
    // Get pointer to start of this row of connectivity and sign change
    uint32_t *rowConnectivity = Connectivity + (RowWords * id_pre);
    uint32_t *rowSignChange = SignChange + (RowWords * id_pre);
    
    // Zero it
    for(int i = 0; i < RowWords; i++) {
        rowConnectivity[i] = 0;
        rowSignChange[i] = 0;
    }
    
    // Set connectivity where there are synapse
    for_each_synapse {
        rowConnectivity[id_post / 32] |= (1 << (id_post % 32));
    }
    """}
}

deep_r_1_model = {
    "params": [("RowWords", "unsigned int")],
    "pre_var_refs": [("NumDormant", "unsigned int")],
    "extra_global_params": [("Connectivity", "uint32_t*"), ("SignChange", "uint32_t*")],
    
    "row_update_code": """
    // Zero dormant counter
    NumDormant = 0;

    // Loop through synapses
    uint32_t *rowSignChange = SignChange + (RowWords * id_pre);
    uint32_t *rowConnectivity = Connectivity + (RowWords * id_pre);
    for_each_synapse {
        // If synapse sign changed
        if(rowSignChange[id_post / 32] & (1 << (id_post % 32))) {
            // Increment dormant counter
            NumDormant++;
            
            // Clear connectivity bit
            rowConnectivity[id_post / 32] &= ~(1 << (id_post % 32));

            // Remove synapse
            remove_synapse();
        }
    }

    // Zero sign change
    for(int i = 0; i < RowWords; i++) {
        rowSignChange[i] = 0;
    }
    """
}

deep_r_2_model = {
    "params": [("RowWords", "unsigned int"), ("MaxRowLength", "unsigned int")],
    "pre_vars": [("NumDormant", "unsigned int"), ("NumActivations", "unsigned int")],
    "extra_global_param_refs": [("Connectivity", "uint32_t*")],
    
    "host_update_code": """
    pullNumDormantFromDevice();
    
    // Count total dormant and zero activations
    size_t numRemainingDormant = 0;
    for(unsigned int i = 0; i < num_pre; i++) {
        numRemainingDormant += NumDormant[i];
        NumActivations[i] = 0;
    }
    
    // Loop through all rows but last
    size_t numTotalPaddingSynapses = (MaxRowLength * num_pre) - num_synapses;
    for(unsigned int i = 0; i < (num_pre - 1); i++) {
        const unsigned int numRowPaddingSynapses = MaxRowLength - row_length[i];
        if(numRowPaddingSynapses > 0 && numTotalPaddingSynapses > 0) {
            const scalar prob = (scalar)numRowPaddingSynapses / numTotalPaddingSynapses;
            
            const unsigned int numRowActivations = min(numRowPaddingSynapses,
                                                       gennrand_binomial(numRemainingDormant, prob));
            NumActivations[i] = numRowActivations;
            
            numRemainingDormant -= numRowActivations
            numTotalPaddingSynapses -= numRowPaddingSynapses;
        }
    }
    
    // Put remaining dormant synapses in last row
    NumActivations[num_pre - 1] = numRemainingDormant;
    
    pushNumActivationsToDevice();
    """,
    
    "row_update_code": """
    // Loop through synapses to activate
    uint32_t *rowConnectivity = Connectivity + (RowWords * id_pre);
    for(unsigned int i = 0; i < NumActivations; i++) {
        while(true) {
            // Pick a random synapse to activate
            const unsigned int j = genn_rand() % num_post;
            
            if(!(rowConnectivity[j / 32] & (1 << (j % 32)))) {
                add_synapse(j);
                rowConnectivity[j / 32] |= (1 << (j % 32));
                break;
            }
        }
    }
    """)
