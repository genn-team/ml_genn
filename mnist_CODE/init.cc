#include "definitionsInternal.h"
void initialize() {
     {
        uint32_t seedData[std::mt19937::state_size];
        std::random_device seedSource;
         {
            for(int i = 0; i < std::mt19937::state_size; i++) {
                seedData[i] = seedSource();
            }
        }
        std::seed_seq seeds(std::begin(seedData), std::end(seedData));
        rng.seed(seeds);
    }
    
    // ------------------------------------------------------------------------
    // Remote neuron groups
    
    // ------------------------------------------------------------------------
    // Local neuron groups
    // neuron group if1
     {
        glbSpkCntif1[0] = 0;
        for (unsigned i = 0; i < 128; i++) {
            glbSpkif1[i] = 0;
        }
         {
            for (unsigned i = 0; i < 128; i++) {
                const scalar scale = (-5.00000000000000000e+01f) - (-6.00000000000000000e+01f);
                Vmemif1[i] = (-6.00000000000000000e+01f) + (standardUniformDistribution(rng) * scale);
            }
        }
         {
            for (unsigned i = 0; i < 128; i++) {
                SpikeNumberif1[i] = (0.00000000000000000e+00f);
            }
        }
        for (unsigned i = 0; i < 128; i++) {
            inSyninput_pop[i] = 0.000000f;
        }
        // current source variables
    }
    // neuron group if2
     {
        glbSpkCntif2[0] = 0;
        for (unsigned i = 0; i < 10; i++) {
            glbSpkif2[i] = 0;
        }
         {
            for (unsigned i = 0; i < 10; i++) {
                const scalar scale = (-5.00000000000000000e+01f) - (-6.00000000000000000e+01f);
                Vmemif2[i] = (-6.00000000000000000e+01f) + (standardUniformDistribution(rng) * scale);
            }
        }
         {
            for (unsigned i = 0; i < 10; i++) {
                SpikeNumberif2[i] = (0.00000000000000000e+00f);
            }
        }
        for (unsigned i = 0; i < 10; i++) {
            inSynsyn12[i] = 0.000000f;
        }
        // current source variables
    }
    // neuron group poisson_pop
     {
        glbSpkCntpoisson_pop[0] = 0;
        for (unsigned i = 0; i < 784; i++) {
            glbSpkpoisson_pop[i] = 0;
        }
         {
            for (unsigned i = 0; i < 784; i++) {
                timeStepToSpikepoisson_pop[i] = (0.00000000000000000e+00f);
            }
        }
         {
            for (unsigned i = 0; i < 784; i++) {
                isipoisson_pop[i] = (0.00000000000000000e+00f);
            }
        }
        // current source variables
    }
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    // synapse group input_pop
     {
        for(unsigned int i = 0; i < 784; i++) {
        }
    }
    // synapse group syn12
     {
        for(unsigned int i = 0; i < 128; i++) {
        }
    }
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
}

void initializeSparse() {
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
}
