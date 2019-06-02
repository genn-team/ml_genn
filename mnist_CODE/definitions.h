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
EXPORT_VAR std::mt19937 rng;
EXPORT_VAR std::uniform_real_distribution<float> standardUniformDistribution;
EXPORT_VAR std::normal_distribution<float> standardNormalDistribution;
EXPORT_VAR std::exponential_distribution<float> standardExponentialDistribution;

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
#define spikeCount_if0 glbSpkCntif0[0]
#define spike_if0 glbSpkif0
#define glbSpkShiftif0 0

EXPORT_VAR unsigned int* glbSpkCntif0;
EXPORT_VAR unsigned int* glbSpkif0;
EXPORT_VAR scalar* Vmemif0;
EXPORT_VAR unsigned int* SpikeNumberif0;
// current source variables
EXPORT_VAR scalar* magnitudecs;
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

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynsyn01;
EXPORT_VAR float* inSynsyn12;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* gsyn01;
EXPORT_VAR scalar* gsyn12;

EXPORT_FUNC void pushif0SpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullif0SpikesFromDevice();
EXPORT_FUNC void pushif0CurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullif0CurrentSpikesFromDevice();
EXPORT_FUNC void pushVmemif0ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVmemif0FromDevice();
EXPORT_FUNC void pushSpikeNumberif0ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSpikeNumberif0FromDevice();
EXPORT_FUNC void pushif0StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullif0StateFromDevice();
EXPORT_FUNC void pushmagnitudecsToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmagnitudecsFromDevice();
EXPORT_FUNC void pushcsStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullcsStateFromDevice();
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
EXPORT_FUNC void pushgsyn01ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgsyn01FromDevice();
EXPORT_FUNC void pushinSynsyn01ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn01FromDevice();
EXPORT_FUNC void pushsyn01StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn01StateFromDevice();
EXPORT_FUNC void pushgsyn12ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgsyn12FromDevice();
EXPORT_FUNC void pushinSynsyn12ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsyn12FromDevice();
EXPORT_FUNC void pushsyn12StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsyn12StateFromDevice();
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
