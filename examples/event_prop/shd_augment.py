import numpy as np
import matplotlib.pyplot as plt
import mnist

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense,FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic import DiskCachedDataset
from tonic.datasets import SHD

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

from ml_genn.compilers.event_prop_compiler import default_params

NUM_HIDDEN = 256
BATCH_SIZE = 32
NUM_EPOCHS = 300
AUGMENT_SHIFT_SD = 2
DT = 1.0
TRAIN = True
KERNEL_PROFILING = True

# Class implementing simple augmentation where all events
# in example are shifted in space by normally-distributed value
class Shift:
    def __init__(self, sd, sensor_size):
        self.sd = sd
        self.sensor_size = sensor_size

    def __call__(self, events: np.ndarray) -> np.ndarray:
        # Shift events
        events["x"] = events["x"] + np.random.normal(scale=self.sd)
        
        # Delete out of bound events
        events = np.delete(
            events,
            np.where(
                (events["x"] < 0) | (events["x"] >= self.sensor_size[0])))
        return events

# Get dataset
dataset = SHD(save_to="../data", train=TRAIN)

# If we're training
if TRAIN:
    # Loop through dataset
    max_spikes = 0
    latest_spike_time = 0
    raw_dataset = []
    for i in range(len(dataset)):
        events, label = dataset[i]
        
        # Add raw events and label to list
        raw_dataset.append((events, label))
        
        # Calculate max spikes and max times
        max_spikes = max(max_spikes, len(events))
        latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)

# Otherwise
else:
    # Preprocess dataset directly
    spikes = []
    labels = []
    for i in range(len(dataset)):
        events, label = dataset[i]
        spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                              dataset.sensor_size))
        labels.append(label)

    # Determine max spikes and latest spike time
    max_spikes = calc_max_spikes(spikes)
    latest_spike_time = calc_latest_spike_time(spikes)

print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)

serialiser = Numpy("shd_augment_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output)

    # Connections
    Connection(input, hidden, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02)),
               Exponential(5.0))
    Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))

max_example_timesteps = int(np.ceil(latest_spike_time / DT))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 reg_lambda_upper=4e-09, reg_lambda_lower=4e-09, 
                                 reg_nu_upper=14, max_spikes=1500, 
                                 optimiser=Adam(0.001), batch_size=BATCH_SIZE, 
                                 kernel_profiling=KERNEL_PROFILING)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Loop through epochs
        start_time = perf_counter()
        callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        augmentation = Shift(AUGMENT_SHIFT_SD, dataset.sensor_size)
        for e in range(NUM_EPOCHS):
            # Apply augmentation to events and preprocess
            spikes = []
            labels = []
            for events, label in raw_dataset:
                spikes.append(preprocess_tonic_spikes(augmentation(events), dataset.ordering,
                                                      dataset.sensor_size))
                labels.append(label)
            
            # Train epoch
            metrics, _  = compiled_net.train({input: spikes},
                                             {output: labels},
                                             start_epoch=e, num_epochs=1, 
                                             shuffle=True, callbacks=callbacks)

        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")

        if KERNEL_PROFILING:
            print(f"Neuron update time = {compiled_net.genn_model.neuron_update_time}")
            print(f"Presynaptic update time = {compiled_net.genn_model.presynaptic_update_time}")
            print(f"Gradient batch reduce time = {compiled_net.genn_model.get_custom_update_time('GradientBatchReduce')}")
            print(f"Gradient learn time = {compiled_net.genn_model.get_custom_update_time('GradientLearn')}")
            print(f"Reset time = {compiled_net.genn_model.get_custom_update_time('Reset')}")
            print(f"Softmax1 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax1')}")
            print(f"Softmax2 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax2')}")
            print(f"Softmax3 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax3')}")
else:
    # Load network state from final checkpoint
    network.load((NUM_EPOCHS - 1,), serialiser)

    compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                 reset_in_syn_between_batches=True,
                                 batch_size=BATCH_SIZE)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels})
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
