#include "definitionsInternal.h"
#include "supportCode.h"

void updateNeurons(float t) {
    // neuron group if0
     {
        glbSpkCntif0[0] = 0;
        
        for(unsigned int i = 0; i < 784; i++) {
            scalar lVmem = Vmemif0[i];
            unsigned int lSpikeNumber = SpikeNumberif0[i];
            
            float Isyn = 0;
            // current source cs
             {
                scalar lcsmagnitude = magnitudecs[i];
                
                Isyn += lcsmagnitude;
                
                magnitudecs[i] = lcsmagnitude;
            }
            // test whether spike condition was fulfilled previously
            const bool oldSpike= (lVmem >= (-5.50000000000000000e+01f));
            // calculate membrane potential
            
            lVmem += Isyn*(DT / (1.00000000000000000e+00f));
            //printf("Vmem: %f, Isyn: %f, SpikeNumber: %d", lVmem,Isyn,lSpikeNumber);
            
            // test for and register a true spike
            if ((lVmem >= (-5.50000000000000000e+01f)) && !(oldSpike)) {
                glbSpkif0[glbSpkCntif0[0]++] = i;
                // spike reset code
                
                lVmem = (-6.00000000000000000e+01f); 
                lSpikeNumber += 1;
                
            }
            Vmemif0[i] = lVmem;
            SpikeNumberif0[i] = lSpikeNumber;
        }
    }
    // neuron group if1
     {
        glbSpkCntif1[0] = 0;
        
        for(unsigned int i = 0; i < 128; i++) {
            scalar lVmem = Vmemif1[i];
            unsigned int lSpikeNumber = SpikeNumberif1[i];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynsyn01 = inSynsyn01[i];
            Isyn += linSynsyn01; linSynsyn01 = 0;
            // test whether spike condition was fulfilled previously
            const bool oldSpike= (lVmem >= (-5.50000000000000000e+01f));
            // calculate membrane potential
            
            lVmem += Isyn*(DT / (1.00000000000000000e+00f));
            //printf("Vmem: %f, Isyn: %f, SpikeNumber: %d", lVmem,Isyn,lSpikeNumber);
            
            // test for and register a true spike
            if ((lVmem >= (-5.50000000000000000e+01f)) && !(oldSpike)) {
                glbSpkif1[glbSpkCntif1[0]++] = i;
                // spike reset code
                
                lVmem = (-6.00000000000000000e+01f); 
                lSpikeNumber += 1;
                
            }
            Vmemif1[i] = lVmem;
            SpikeNumberif1[i] = lSpikeNumber;
            // the post-synaptic dynamics
            
            inSynsyn01[i] = linSynsyn01;
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
            const bool oldSpike= (lVmem >= (-5.50000000000000000e+01f));
            // calculate membrane potential
            
            lVmem += Isyn*(DT / (1.00000000000000000e+00f));
            //printf("Vmem: %f, Isyn: %f, SpikeNumber: %d", lVmem,Isyn,lSpikeNumber);
            
            // test for and register a true spike
            if ((lVmem >= (-5.50000000000000000e+01f)) && !(oldSpike)) {
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
}
