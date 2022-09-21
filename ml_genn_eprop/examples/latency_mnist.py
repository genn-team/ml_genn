import numpy as np
import matplotlib.pyplot as plt
import mnist

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn_eprop import EPropCompiler

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                log_latency_encode_data)

NUM_INPUT = 784
NUM_HIDDEN = 100
NUM_OUTPUT = 16

training_labels = mnist.train_labels()
training_spikes = log_latency_encode_data(mnist.train_images(), 20.0, 51, 100)

network = SequentialNetwork()
with network:
    # Populations
    input = InputLayer(SpikeInput(max_spikes=128 * calc_max_spikes(training_spikes)),
                                  NUM_INPUT)
    hidden = Layer(Dense(Normal(sd=1.0 / np.sqrt(NUM_INPUT))),
                   LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0,
                                      tau_refrac=5.0, 
                                      relative_reset=True,
                                      integrate_during_refrac=True),
                   NUM_HIDDEN)
    output = Layer(Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))),
                   LeakyIntegrate(tau_mem=20.0, softmax=True, readout="sum_var"),
                   NUM_OUTPUT)

max_example_time = calc_latest_spike_time(training_spikes)
compiler = EPropCompiler(example_timesteps=int(np.ceil(max_example_time)),
                         losses="sparse_categorical_crossentropy",
                         optimiser="adam", batch_size=128)
compiled_net = compiler.compile(network)

with compiled_net:
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    metrics, cb_data  = compiled_net.train({input: training_spikes},
                                           {output: training_labels},
                                           num_epochs=50, shuffle=True)
    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
    print(f"Time = {end_time - start_time}s")