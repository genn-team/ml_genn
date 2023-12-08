import numpy as np

from ml_genn import Connection, Population, Network
from ml_genn.callbacks import Checkpoint, VarRecorder
from ml_genn.communicators import MPI
from ml_genn.compilers import EPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import (LeakyIntegrate, AdaptiveLeakyIntegrateFire,
                             SpikeInput)
from ml_genn.serialisers import Numpy
from tonic.datasets import SHD

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

from ml_genn.compilers.eprop_compiler import default_params
from pygenn.cuda_backend import DeviceSelect

NUM_HIDDEN = 256
BATCH_SIZE = 512
NUM_EPOCHS = 50

TRAIN = True
KERNEL_PROFILING = False

# Get SHD dataset
dataset = SHD(save_to='../data', train=True)

if TRAIN:
    communicator = MPI()
    print(f"Training on rank {communicator.rank} / {communicator.num_ranks}")

    data_range = range(communicator.rank, len(dataset), communicator.num_ranks)
else:
    data_range = range(len(dataset))

# Preprocess
spikes = []
labels = []
for i in data_range:
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

serialiser = Numpy("shd_mpi_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    hidden = Population(AdaptiveLeakyIntegrateFire(v_thresh=0.6,
                                                   tau_refrac=5.0),
                        NUM_HIDDEN)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="sum_var"), num_output)

    # Connections
    Connection(input, hidden, Dense(Normal(sd=1.0 / np.sqrt(num_input))))
    Connection(hidden, hidden, Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))))
    Connection(hidden, output, Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))))

if TRAIN:
    compiler = EPropCompiler(example_timesteps=int(np.ceil(latest_spike_time)),
                            losses="sparse_categorical_crossentropy",
                            optimiser="adam", batch_size=BATCH_SIZE,
                            communicator=communicator,
                            device_select_method=DeviceSelect.MANUAL)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on SHD
        start_time = perf_counter()
        if communicator.rank == 0:
            callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        else:
            callbacks = []
        metrics, _  = compiled_net.train({input: spikes},
                                            {output: labels},
                                            num_epochs=50,
                                            callbacks=callbacks,
                                            shuffle=True)
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
else:
    # Load network state from final checkpoint
    network.load((NUM_EPOCHS - 1,), serialiser)

    compiler = InferenceCompiler(evaluate_timesteps=int(np.ceil(latest_spike_time)),
                                 reset_vars_between_batches=False, batch_size=BATCH_SIZE)
    compiled_net = compiler.compile(network)

    with compiled_net:
        compiled_net.evaluate({input: spikes}, {output: labels})

        # Evaluate model on numpy dataset
        start_time = perf_counter()
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels})
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
