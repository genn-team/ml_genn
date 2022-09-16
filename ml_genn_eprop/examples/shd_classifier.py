import numpy as np

from ml_genn import Connection, Population, Network
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import (LeakyIntegrate, AdaptiveLeakyIntegrateFire,
                             SpikeInput)
from ml_genn_eprop import EPropCompiler
from tonic.datasets import SHD

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

NUM_HIDDEN = 256
BATCH_SIZE = 128

# Get SHD dataset
dataset = SHD(save_to='./data', train=True)

# Preprocess
spikes = []
labels = []
for events, label in dataset:
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
padded_num_output = int(2**(np.ceil(np.log2(num_output))))

network = Network()
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    hidden = Population(AdaptiveLeakyIntegrateFire(v_thresh=0.6,
                                                   tau_refrac=5.0,
                                                   relative_reset=True,
                                                   integrate_during_refrac=True),
                        NUM_HIDDEN)
    output = Population(LeakyIntegrate(tau_mem=20.0, output="sum_var", softmax=True),
                        padded_num_output)
    
    # Connections
    Connection(input, hidden, Dense(Normal(sd=1.0 / np.sqrt(num_input))))
    Connection(hidden, hidden, Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))))
    Connection(hidden, output, Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))))

compiler = EPropCompiler(example_timesteps=int(np.ceil(latest_spike_time)),
                         losses="sparse_categorical_crossentropy", 
                         optimiser="adam")
compiled_net = compiler.compile(network)

with compiled_net:
    # Evaluate model on SHD
    start_time = perf_counter()
    metrics, cb_data  = compiled_net.train({input: spikes},
                                           {output: labels},
                                           num_epochs=50)
    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
    print(f"Time = {end_time - start_time}s")