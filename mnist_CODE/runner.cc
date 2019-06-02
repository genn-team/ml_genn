#include "definitionsInternal.h"

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
float t;
std::mt19937 rng;
std::uniform_real_distribution<float> standardUniformDistribution(0.000000f, 1.000000f);
std::normal_distribution<float> standardNormalDistribution(0.000000f, 1.000000f);
std::exponential_distribution<float> standardExponentialDistribution(1.000000f);

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double neuronUpdateTime;
double initTime;
double presynapticUpdateTime;
double postsynapticUpdateTime;
double synapseDynamicsTime;
double initSparseTime;
// ------------------------------------------------------------------------
// remote neuron groups
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntif0;
unsigned int* glbSpkif0;
scalar* Vmemif0;
unsigned int* SpikeNumberif0;
// current source variables
scalar* magnitudecs;
unsigned int* glbSpkCntif1;
unsigned int* glbSpkif1;
scalar* Vmemif1;
unsigned int* SpikeNumberif1;
unsigned int* glbSpkCntif2;
unsigned int* glbSpkif2;
scalar* Vmemif2;
unsigned int* SpikeNumberif2;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
float* inSynsyn01;
float* inSynsyn12;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
scalar* gsyn01;
scalar* gsyn12;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushif0SpikesToDevice(bool uninitialisedOnly) {
}

void pushif0CurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVmemif0ToDevice(bool uninitialisedOnly) {
}

void pushSpikeNumberif0ToDevice(bool uninitialisedOnly) {
}

void pushif0StateToDevice(bool uninitialisedOnly) {
    pushVmemif0ToDevice(uninitialisedOnly);
    pushSpikeNumberif0ToDevice(uninitialisedOnly);
}

void pushmagnitudecsToDevice(bool uninitialisedOnly) {
}

void pushcsStateToDevice(bool uninitialisedOnly) {
    pushmagnitudecsToDevice(uninitialisedOnly);
}

void pushif1SpikesToDevice(bool uninitialisedOnly) {
}

void pushif1CurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVmemif1ToDevice(bool uninitialisedOnly) {
}

void pushSpikeNumberif1ToDevice(bool uninitialisedOnly) {
}

void pushif1StateToDevice(bool uninitialisedOnly) {
    pushVmemif1ToDevice(uninitialisedOnly);
    pushSpikeNumberif1ToDevice(uninitialisedOnly);
}

void pushif2SpikesToDevice(bool uninitialisedOnly) {
}

void pushif2CurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVmemif2ToDevice(bool uninitialisedOnly) {
}

void pushSpikeNumberif2ToDevice(bool uninitialisedOnly) {
}

void pushif2StateToDevice(bool uninitialisedOnly) {
    pushVmemif2ToDevice(uninitialisedOnly);
    pushSpikeNumberif2ToDevice(uninitialisedOnly);
}

void pushgsyn01ToDevice(bool uninitialisedOnly) {
}

void pushinSynsyn01ToDevice(bool uninitialisedOnly) {
}

void pushsyn01StateToDevice(bool uninitialisedOnly) {
    pushgsyn01ToDevice(uninitialisedOnly);
    pushinSynsyn01ToDevice(uninitialisedOnly);
}

void pushgsyn12ToDevice(bool uninitialisedOnly) {
}

void pushinSynsyn12ToDevice(bool uninitialisedOnly) {
}

void pushsyn12StateToDevice(bool uninitialisedOnly) {
    pushgsyn12ToDevice(uninitialisedOnly);
    pushinSynsyn12ToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullif0SpikesFromDevice() {
}

void pullif0CurrentSpikesFromDevice() {
}

void pullVmemif0FromDevice() {
}

void pullSpikeNumberif0FromDevice() {
}

void pullif0StateFromDevice() {
    pullVmemif0FromDevice();
    pullSpikeNumberif0FromDevice();
}

void pullmagnitudecsFromDevice() {
}

void pullcsStateFromDevice() {
    pullmagnitudecsFromDevice();
}

void pullif1SpikesFromDevice() {
}

void pullif1CurrentSpikesFromDevice() {
}

void pullVmemif1FromDevice() {
}

void pullSpikeNumberif1FromDevice() {
}

void pullif1StateFromDevice() {
    pullVmemif1FromDevice();
    pullSpikeNumberif1FromDevice();
}

void pullif2SpikesFromDevice() {
}

void pullif2CurrentSpikesFromDevice() {
}

void pullVmemif2FromDevice() {
}

void pullSpikeNumberif2FromDevice() {
}

void pullif2StateFromDevice() {
    pullVmemif2FromDevice();
    pullSpikeNumberif2FromDevice();
}

void pullgsyn01FromDevice() {
}

void pullinSynsyn01FromDevice() {
}

void pullsyn01StateFromDevice() {
    pullgsyn01FromDevice();
    pullinSynsyn01FromDevice();
}

void pullgsyn12FromDevice() {
}

void pullinSynsyn12FromDevice() {
}

void pullsyn12StateFromDevice() {
    pullgsyn12FromDevice();
    pullinSynsyn12FromDevice();
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushif0StateToDevice(uninitialisedOnly);
    pushif1StateToDevice(uninitialisedOnly);
    pushif2StateToDevice(uninitialisedOnly);
    pushcsStateToDevice(uninitialisedOnly);
    pushsyn01StateToDevice(uninitialisedOnly);
    pushsyn12StateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
}

void copyStateFromDevice() {
    pullif0StateFromDevice();
    pullif1StateFromDevice();
    pullif2StateFromDevice();
    pullcsStateFromDevice();
    pullsyn01StateFromDevice();
    pullsyn12StateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullif0CurrentSpikesFromDevice();
    pullif1CurrentSpikesFromDevice();
    pullif2CurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // remote neuron groups
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    glbSpkCntif0 = new unsigned int[1];
    glbSpkif0 = new unsigned int[784];
    Vmemif0 = new scalar[784];
    SpikeNumberif0 = new unsigned int[784];
    // current source variables
    magnitudecs = new scalar[784];
    glbSpkCntif1 = new unsigned int[1];
    glbSpkif1 = new unsigned int[128];
    Vmemif1 = new scalar[128];
    SpikeNumberif1 = new unsigned int[128];
    glbSpkCntif2 = new unsigned int[1];
    glbSpkif2 = new unsigned int[10];
    Vmemif2 = new scalar[10];
    SpikeNumberif2 = new unsigned int[10];
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSynsyn01 = new float[128];
    inSynsyn12 = new float[10];
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    gsyn01 = new scalar[100352];
    gsyn12 = new scalar[1280];
    
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // remote neuron groups
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    delete[] glbSpkCntif0;
    delete[] glbSpkif0;
    delete[] Vmemif0;
    delete[] SpikeNumberif0;
    // current source variables
    delete[] magnitudecs;
    delete[] glbSpkCntif1;
    delete[] glbSpkif1;
    delete[] Vmemif1;
    delete[] SpikeNumberif1;
    delete[] glbSpkCntif2;
    delete[] glbSpkif2;
    delete[] Vmemif2;
    delete[] SpikeNumberif2;
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    delete[] inSynsyn01;
    delete[] inSynsyn12;
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    delete[] gsyn01;
    delete[] gsyn12;
    
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t);
    iT++;
    t = iT*DT;
}

