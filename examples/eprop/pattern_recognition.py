import numpy as np
import matplotlib.pyplot as plt

from ml_genn import Connection, Population, Network
from ml_genn.callbacks import (OptimiserParamSchedule, SpikeRecorder,
                               VarRecorder)
from ml_genn.compilers import EPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam

from time import perf_counter
from ml_genn.utils.data import preprocess_spikes

from ml_genn.compilers.eprop_compiler import default_params

NUM_INPUT = 20
NUM_HIDDEN = 256
NUM_OUTPUT = 3

NUM_FREQ_COMP = 3

IN_GROUP_SIZE = 4
IN_ACTIVE_ISI = 10
IN_ACTIVE_INTERVAL = 200

NUM_EPOCHS = 1000
DISPLAY_EPOCH_INTERVAL = 100

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
y_star = np.zeros((1000, NUM_OUTPUT))
for i in range(NUM_OUTPUT):
    for c in range(NUM_FREQ_COMP):
        y_star[:, i] += output_ampl[i, c] * sinusoids[:, c] + output_phase[i, c]


# Determine which group each input neuron is in
in_group = np.arange(NUM_INPUT, dtype=int) // IN_GROUP_SIZE

# Fill matrix rows with base spike times
num_spikes_per_neuron = IN_ACTIVE_INTERVAL // IN_ACTIVE_ISI
in_spike_times = np.empty((NUM_INPUT, num_spikes_per_neuron), dtype=np.float32)
in_spike_times[:] = np.arange(0, IN_ACTIVE_INTERVAL, IN_ACTIVE_ISI)

# Shift each spike time by group start
in_spike_times += np.reshape(in_group * IN_ACTIVE_INTERVAL, (NUM_INPUT, 1))

# Create matching array of IDs
in_spike_ids = np.repeat(np.arange(NUM_INPUT), num_spikes_per_neuron)

# Pre-process spikes
in_spikes = preprocess_spikes(in_spike_times.flatten(), in_spike_ids, NUM_INPUT)

network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=len(in_spike_ids)),
                                  NUM_INPUT, record_spikes=True)
    hidden = Population(LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0,
                                           tau_refrac=5.0),
                        NUM_HIDDEN, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="var"),
                        NUM_OUTPUT)
    
    # Connections
    Connection(input, hidden, Dense(Normal(sd=1.0 / np.sqrt(NUM_INPUT))))
    Connection(hidden, hidden, Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))))
    Connection(hidden, output, Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))))

compiler = EPropCompiler(example_timesteps=1000, losses="mean_square_error",
                         optimiser=Adam(0.003), c_reg=3.0)
compiled_net = compiler.compile(network)

with compiled_net:
    def alpha_schedule(epoch, alpha):
        if (epoch % 200) == 0 and epoch != 0:
            return alpha * 0.7
        else:
            return alpha

    # Evaluate model on numpy dataset
    start_time = perf_counter()
    callbacks = ["batch_progress_bar", 
                 VarRecorder(output, "v", key="output_v"),
                 SpikeRecorder(input, key="input_spikes"),
                 SpikeRecorder(hidden, key="hidden_spikes"),
                 OptimiserParamSchedule("alpha", alpha_schedule)]
    metrics, cb_data  = compiled_net.train({input: [in_spikes]},
                                           {output: [y_star]},
                                           num_epochs=NUM_EPOCHS,
                                           callbacks=callbacks)
    end_time = perf_counter()
    print(f"Time = {end_time - start_time}s")
    
    num_display_epochs = NUM_EPOCHS // DISPLAY_EPOCH_INTERVAL
    fig, axes = plt.subplots(NUM_FREQ_COMP + 2, num_display_epochs, sharex="col", sharey="row")
    for i in range(num_display_epochs):
        epoch = i * DISPLAY_EPOCH_INTERVAL
        error = []
        for c in range(NUM_FREQ_COMP):
            y = cb_data["output_v"][epoch][:,c]
            error.append(y - y_star[:,c])
            mse = np.sum(error[-1] * error[-1]) / len(error[-1])
            axes[c,i].set_title(f"Y{c} (MSE={mse:.2f})")
            axes[c,i].plot(y)
            axes[c,i].plot(y_star[:,c], linestyle="--")
        
        axes[NUM_FREQ_COMP,i].scatter(cb_data["input_spikes"][0][epoch],
                                      cb_data["input_spikes"][1][epoch], s=2)
        axes[NUM_FREQ_COMP + 1,i].scatter(cb_data["hidden_spikes"][0][epoch],
                                          cb_data["hidden_spikes"][1][epoch], s=2)
        
        error = np.hstack(error)
        total_mse = np.sum(error * error) / len(error)
        print(f"{i}: Total MSE: {total_mse}")
    axes[0,0].set_ylabel("Input spikes")
    axes[0,1].set_ylabel("Y")
    axes[0,2].set_ylabel("E")
    plt.show()
