import numpy as np

from ml_genn import Connection, Population, Network
from ml_genn.callbacks import VarRecorder
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn_eprop import EPropCompiler

from time import perf_counter
from ml_genn.utils.data import preprocess_spikes

NUM_INPUT = 20
NUM_HIDDEN = 256
NUM_OUTPUT = 3

NUM_FREQ_COMP = 3

IN_GROUP_SIZE = 4
IN_ACTIVE_ISI = 10
IN_ACTIVE_INTERVAL = 200

# Pick random phases and outputs for the three frequency components 
# components used to generate patterns for each output neuron
output_phase = np.random.random((NUM_OUTPUT, NUM_FREQ_COMP)) * 2.0 * np.pi
output_ampl = 0.5 + (np.random.random((NUM_OUTPUT, NUM_FREQ_COMP)) * (2.0 - 0.5))

# Convert frequencies of each component into row vector
freq = [2.0, 3.0, 5.0]
freq_radians = np.multiply(freq, 1.0 * np.pi / 1000.0)

# Calculate sinusoid of each frequency for 
sinusoids = np.sin(np.outer(np.arange(1000), freq_radians))

# Calculate Y* target
y_star = np.zeros((1000, 1, NUM_OUTPUT))
for i in range(NUM_OUTPUT):
    for c in range(NUM_FREQ_COMP):
        y_star[:, 0, i] += output_ampl[i, c] * sinusoids[:, c] + output_phase[i, c]


# Determine which group each input neuron is in
in_group = np.arange(NUM_INPUT, dtype=int) // IN_GROUP_SIZE

# Fill matrix rows with base spike times
num_spikes_per_neuron = IN_ACTIVE_INTERVAL // IN_ACTIVE_ISI
in_spike_times = np.empty((NUM_INPUT, num_spikes_per_neuron))
in_spike_times[:] = np.arange(0, IN_ACTIVE_INTERVAL, IN_ACTIVE_ISI)

# Shift each spike time by group start
in_spike_times += np.reshape(in_group * IN_ACTIVE_INTERVAL, (NUM_INPUT, 1))

# Create matching array of IDs
in_spike_ids = np.repeat(np.arange(NUM_INPUT), num_spikes_per_neuron)

# Pre-process spikes
in_spikes = preprocess_spikes(in_spike_times.flatten(), in_spike_ids, NUM_INPUT)

network = Network()
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=len(in_spike_ids)),
                                  NUM_INPUT)
    hidden = Population(LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0,
                                           tau_refrac=5.0, 
                                           relative_reset=True,
                                           integrate_during_refrac=True),
                        NUM_HIDDEN)
    output = Population(LeakyIntegrate(tau_mem=20.0, output="var"),
                        NUM_OUTPUT)
    
    # Connections
    Connection(input, hidden, Dense(Normal(sd=1.0 / np.sqrt(NUM_INPUT))))
    Connection(hidden, hidden, Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))))
    Connection(hidden, output, Dense(Normal(sd=1.0 / np.sqrt(NUM_INPUT))))

compiler = EPropCompiler(example_timesteps=1000, losses="mean_square_error",
                         optimiser=Adam(), c_reg=3.0)
compiled_net = compiler.compile(network)

with compiled_net:
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    metrics, _ = compiled_net.train({input: in_spikes},
                                    {output: y_star},
                                    num_epochs=1000,
                                    metrics="mean_square_error")
    end_time = perf_counter()
    print(f"Accuracy = {metrics[output].result}")
    print(f"Time = {end_time - start_time}s")
