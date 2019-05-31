#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <chrono>
#include <iostream>
#include <random>

// Standard C includes
#include <cmath>
#include <cstdint>
#include <cstring>
#define DT 1.000000f
typedef float scalar;
#define SCALAR_MIN 1.175494351e-38f
#define SCALAR_MAX 3.402823466e+38f

#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double initTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
EXPORT_VAR double initSparseTime;
// ------------------------------------------------------------------------
// remote neuron groups
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_if1 glbSpkCntif1[0]
#define spike_if1 glbSpkif1
#define glbSpkShiftif1 0

EXPORT_VAR unsigned int* glbSpkCntif1;
EXPORT_VAR unsigned int* glbSpkif1;
EXPORT_VAR scalar* Vmemif1;
EXPORT_VAR unsigned int* SpikeNumberif1;
#define spikeCount_if2 glbSpkCntif2[0]
#define spike_if2 glbSpkif2
#define glbSpkShiftif2 0

EXPORT_VAR unsigned int* glbSpkCntif2;
EXPORT_VAR unsigned int* glbSpkif2;
EXPORT_VAR scalar* Vmemif2;
EXPORT_VAR unsigned int* SpikeNumberif2;
#define spikeCount_poisson_pop glbSpkCntpoisson_pop[0]
#define spike_poisson_pop glbSpkpoisson_pop
#define glbSpkShiftpoisson_pop 0

EXPORT_VAR unsigned int* glbSpkCntpoisson_pop;
EXPORT_VAR unsigned int* glbSpkpoisson_pop;
EXPORT_VAR scalar* timeStepToSpikepoisson_pop;
EXPORT_VAR scalar* isipoisson_pop;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSyninput_pop;
EXPORT_VAR float* inSynsyn21;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* ginput_pop;
EXPORT_VAR scalar* gsyn21;

EXPORT_FUNC void pushif1SpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullif1SpikesFromDevice();
EXPORT_FUNC void pushif1CurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullif1CurrentSpikesFromDevice();
EXPORT_FUNC void pushVmemif1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVmemif1FromDevice();
EXPORT_FUNC void pushSpikeNumberif1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSpikeNumberif1FromDevice();
EXPORT_FUNC void pushif1StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullif1StateFromDevice();
EXPORT_FUNC void pushif2SpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullif2SpikesFromDevice();
EXPORT_FUNC void pushif2CurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullif2CurrentSpikesFromDevice();
EXPORT_FUNC void pushVmemif2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVmemif2FromDevice();
EXPORT_FUNC void pushSpikeNumberif2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSpikeNumberif2FromDevice();
EXPORT_FUNC void pushif2StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullif2StateFromDevice();
EXPORT_FUNC void pushpoisson_popSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpoisson_popSpikesFromDevice();
EXPORT_FUNC void pushpoisson_popCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpoisson_popCurrentSpikesFromDevice();
EXPORT_FUNC void pushtimeStepToSpikepoisson_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltimeStepToSpikepoisson_popFromDevice();
EXPORT_FUNC void pushisipoisson_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullisipoisson_popFromDevice();
EXPORT_FUNC void pushpoisson_popStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpoisson_popStateFromDevice();
EXPORT_FUNC void pushginput_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullginput_popFromDevice();
EXPORT_FUNC void pushinSyninput_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSyninput_popFromDevice();
EXPORT_FUNC void pushinput_popStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinput_popStateFromDevice();
EXPORT_FUNC void pushgsyn21ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgsyn21FromDevice();
EXPORT_FUNC void pushinSynsyn21ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn21FromDevice();
EXPORT_FUNC void pushsyn21StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn21StateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(float t);
EXPORT_FUNC void updateSynapses(float t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
}  // extern "C"
