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
from tonic.datasets import SHD
from tonic.transforms import Compose, CropTime
from typing import Tuple

from dataclasses import dataclass
from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

from ml_genn.compilers.event_prop_compiler import default_params


@dataclass
class EventsToGrid:
    sensor_size: Tuple[int, int, int]
    dt: float

    def __call__(self, events):
        # Tuple of possible axis names
        axes = ("x", "y", "p")

        # Build bin and sample data structures for histogramdd
        bins = []
        sample = []
        for s, a in zip(self.sensor_size, axes):
            if a in events.dtype.names:
                bins.append(np.linspace(0, s, s + 1))
                sample.append(events[a])

        # Add time bins
        bins.append(np.arange(0.0, np.amax(events["t"]) + self.dt, self.dt))
        sample.append(events["t"])

        # Build histogram
        event_hist,_ = np.histogramdd(tuple(sample), tuple(bins))
        new_events = np.where(event_hist > 0)

        # Copy x, y, p data into new structured array
        grid_events = np.empty(len(new_events[0]), dtype=events.dtype)
        for i, a in enumerate(axes):
            if a in grid_events.dtype.names:
                grid_events[a] = new_events[i]

        # Add t, scaling by dt
        grid_events["t"] = new_events[-1] * self.dt
        return grid_events


NUM_HIDDEN = 256
BATCH_SIZE = 32
NUM_EPOCHS = 300
EXAMPLE_TIME = 20.0
DT = 8.0
TRAIN = True
KERNEL_PROFILING = True

# Get SHD dataset, crop each example to 1000ms and ensure grids are aligned to timestep grid
dataset = SHD(save_to="../data", train=TRAIN,
              transform=Compose([CropTime(max=1000.0 * 1000.0), EventsToGrid(SHD.sensor_size, DT * 1000.0)]))


# Preprocess
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

serialiser = Numpy("shd_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var"),
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
                                 optimiser=Adam(0.001), dt=DT, batch_size=BATCH_SIZE,
                                 kernel_profiling=KERNEL_PROFILING)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        metrics, _  = compiled_net.train({input: spikes},
                                         {output: labels},
                                         num_epochs=NUM_EPOCHS, shuffle=True,
                                         callbacks=callbacks)
        compiled_net.save_connectivity((NUM_EPOCHS - 1,), serialiser)

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
                                 dt=DT, batch_size=BATCH_SIZE)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels})
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
