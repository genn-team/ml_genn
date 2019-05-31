#include "definitionsInternal.h"

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
float t;

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
unsigned int* glbSpkCntif1;
unsigned int* glbSpkif1;
scalar* Vmemif1;
unsigned int* SpikeNumberif1;
unsigned int* glbSpkCntif2;
unsigned int* glbSpkif2;
scalar* Vmemif2;
unsigned int* SpikeNumberif2;
unsigned int* glbSpkCntpoisson_pop;
unsigned int* glbSpkpoisson_pop;
scalar* timeStepToSpikepoisson_pop;
scalar* isipoisson_pop;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
float* inSyninput_pop;
float* inSynsyn21;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
scalar* ginput_pop;
scalar* gsyn21;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
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

void pushpoisson_popSpikesToDevice(bool uninitialisedOnly) {
}

void pushpoisson_popCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushtimeStepToSpikepoisson_popToDevice(bool uninitialisedOnly) {
}

void pushisipoisson_popToDevice(bool uninitialisedOnly) {
}

void pushpoisson_popStateToDevice(bool uninitialisedOnly) {
    pushtimeStepToSpikepoisson_popToDevice(uninitialisedOnly);
    pushisipoisson_popToDevice(uninitialisedOnly);
}

void pushginput_popToDevice(bool uninitialisedOnly) {
}

void pushinSyninput_popToDevice(bool uninitialisedOnly) {
}

void pushinput_popStateToDevice(bool uninitialisedOnly) {
    pushginput_popToDevice(uninitialisedOnly);
    pushinSyninput_popToDevice(uninitialisedOnly);
}

void pushgsyn21ToDevice(bool uninitialisedOnly) {
}

void pushinSynsyn21ToDevice(bool uninitialisedOnly) {
}

void pushsyn21StateToDevice(bool uninitialisedOnly) {
    pushgsyn21ToDevice(uninitialisedOnly);
    pushinSynsyn21ToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
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

void pullpoisson_popSpikesFromDevice() {
}

void pullpoisson_popCurrentSpikesFromDevice() {
}

void pulltimeStepToSpikepoisson_popFromDevice() {
}

void pullisipoisson_popFromDevice() {
}

void pullpoisson_popStateFromDevice() {
    pulltimeStepToSpikepoisson_popFromDevice();
    pullisipoisson_popFromDevice();
}

void pullginput_popFromDevice() {
}

void pullinSyninput_popFromDevice() {
}

void pullinput_popStateFromDevice() {
    pullginput_popFromDevice();
    pullinSyninput_popFromDevice();
}

void pullgsyn21FromDevice() {
}

void pullinSynsyn21FromDevice() {
}

void pullsyn21StateFromDevice() {
    pullgsyn21FromDevice();
    pullinSynsyn21FromDevice();
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushif1StateToDevice(uninitialisedOnly);
    pushif2StateToDevice(uninitialisedOnly);
    pushpoisson_popStateToDevice(uninitialisedOnly);
    pushinput_popStateToDevice(uninitialisedOnly);
    pushsyn21StateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
}

void copyStateFromDevice() {
    pullif1StateFromDevice();
    pullif2StateFromDevice();
    pullpoisson_popStateFromDevice();
    pullinput_popStateFromDevice();
    pullsyn21StateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullif1CurrentSpikesFromDevice();
    pullif2CurrentSpikesFromDevice();
    pullpoisson_popCurrentSpikesFromDevice();
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
    glbSpkCntif1 = new unsigned int[1];
    glbSpkif1 = new unsigned int[128];
    Vmemif1 = new scalar[128];
    SpikeNumberif1 = new unsigned int[128];
    glbSpkCntif2 = new unsigned int[1];
    glbSpkif2 = new unsigned int[10];
    Vmemif2 = new scalar[10];
    SpikeNumberif2 = new unsigned int[10];
    glbSpkCntpoisson_pop = new unsigned int[1];
    glbSpkpoisson_pop = new unsigned int[784];
    timeStepToSpikepoisson_pop = new scalar[784];
    isipoisson_pop = new scalar[784];
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSyninput_pop = new float[128];
    inSynsyn21 = new float[10];
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    ginput_pop = new scalar[100352];
    gsyn21 = new scalar[1280];
    
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
    delete[] glbSpkCntif1;
    delete[] glbSpkif1;
    delete[] Vmemif1;
    delete[] SpikeNumberif1;
    delete[] glbSpkCntif2;
    delete[] glbSpkif2;
    delete[] Vmemif2;
    delete[] SpikeNumberif2;
    delete[] glbSpkCntpoisson_pop;
    delete[] glbSpkpoisson_pop;
    delete[] timeStepToSpikepoisson_pop;
    delete[] isipoisson_pop;
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    delete[] inSyninput_pop;
    delete[] inSynsyn21;
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    delete[] ginput_pop;
    delete[] gsyn21;
    
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t);
    iT++;
    t = iT*DT;
}

