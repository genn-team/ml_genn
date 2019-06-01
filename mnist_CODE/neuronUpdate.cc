#include "definitionsInternal.h"
#include "supportCode.h"

void updateNeurons(float t) {
    // neuron group if1
     {
        glbSpkCntif1[0] = 0;
        
        for(unsigned int i = 0; i < 128; i++) {
            scalar lVmem = Vmemif1[i];
            unsigned int lSpikeNumber = SpikeNumberif1[i];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSyninput_pop = inSyninput_pop[i];
            Isyn += linSyninput_pop; linSyninput_pop = 0;
            // test whether spike condition was fulfilled previously
            const bool oldSpike= (lVmem >= (-5.00000000000000000e+01f));
            // calculate membrane potential
            
            lVmem += Isyn*(DT / (1.00000000000000002e-02f));
            printf("%f ",Isyn);
            
            // test for and register a true spike
            if ((lVmem >= (-5.00000000000000000e+01f)) && !(oldSpike)) {
                glbSpkif1[glbSpkCntif1[0]++] = i;
                // spike reset code
                
                lVmem = (-6.00000000000000000e+01f); 
                lSpikeNumber += 1;
                
            }
            Vmemif1[i] = lVmem;
            SpikeNumberif1[i] = lSpikeNumber;
            // the post-synaptic dynamics
            
            inSyninput_pop[i] = linSyninput_pop;
        }
    }
    // neuron group if2
     {
        glbSpkCntif2[0] = 0;
        
        for(unsigned int i = 0; i < 10; i++) {
            scalar lVmem = Vmemif2[i];
            unsigned int lSpikeNumber = SpikeNumberif2[i];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn12 = inSynsyn12[i];
            Isyn += linSynsyn12; linSynsyn12 = 0;
            // test whether spike condition was fulfilled previously
            const bool oldSpike= (lVmem >= (-5.00000000000000000e+01f));
            // calculate membrane potential
            
            lVmem += Isyn*(DT / (1.00000000000000002e-02f));
            printf("%f ",Isyn);
            
            // test for and register a true spike
            if ((lVmem >= (-5.00000000000000000e+01f)) && !(oldSpike)) {
                glbSpkif2[glbSpkCntif2[0]++] = i;
                // spike reset code
                
                lVmem = (-6.00000000000000000e+01f); 
                lSpikeNumber += 1;
                
            }
            Vmemif2[i] = lVmem;
            SpikeNumberif2[i] = lSpikeNumber;
            // the post-synaptic dynamics
            
            inSynsyn12[i] = linSynsyn12;
        }
    }
    // neuron group poisson_pop
     {
        glbSpkCntpoisson_pop[0] = 0;
        
        for(unsigned int i = 0; i < 784; i++) {
            scalar ltimeStepToSpike = timeStepToSpikepoisson_pop[i];
            scalar lisi = isipoisson_pop[i];
            
            float Isyn = 0;
            // current source current_source
             {
                Isyn += (1.00000000000000000e+01f);
                
            }
            // test whether spike condition was fulfilled previously
            const bool oldSpike= (ltimeStepToSpike <= 0.0f);
            // calculate membrane potential
            
            Isyn;
            if(ltimeStepToSpike > 0){
                ltimeStepToSpike -= 1.0f;
            }
            
            // test for and register a true spike
            if ((ltimeStepToSpike <= 0.0f) && !(oldSpike)) {
                glbSpkpoisson_pop[glbSpkCntpoisson_pop[0]++] = i;
                // spike reset code
                
                ltimeStepToSpike += lisi;
                
            }
            timeStepToSpikepoisson_pop[i] = ltimeStepToSpike;
            isipoisson_pop[i] = lisi;
        }
    }
}
