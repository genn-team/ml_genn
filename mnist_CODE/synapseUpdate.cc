#include "definitionsInternal.h"
#include "supportCode.h"

void updateSynapses(float t) {
    // synapse group syn01
     {
        // process presynaptic events: True Spikes
        for (unsigned int i = 0; i < glbSpkCntif0[0]; i++) {
            const unsigned int ipre = glbSpkif0[i];
            for (unsigned int ipost = 0; ipost < 128; ipost++) {
                const unsigned int synAddress = (ipre * 128) + ipost;
                inSynsyn01[ipost] += gsyn01[synAddress];
            }
        }
        
    }
    // synapse group syn12
     {
        // process presynaptic events: True Spikes
        for (unsigned int i = 0; i < glbSpkCntif1[0]; i++) {
            const unsigned int ipre = glbSpkif1[i];
            for (unsigned int ipost = 0; ipost < 10; ipost++) {
                const unsigned int synAddress = (ipre * 10) + ipost;
                inSynsyn12[ipost] += gsyn12[synAddress];
            }
        }
        
    }
}
