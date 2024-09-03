import numpy as np
import matplotlib.pyplot as plt
import mnist

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import (Checkpoint, ConnVarRecorder,
                               SpikeRecorder, VarRecorder)
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense,FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic.datasets import SHD
from tonic.transforms import CropTime

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)


NUM_HIDDEN = 64
BATCH_SIZE = 32
NUM_EPOCHS = 2
DT = 1.0
TRAIN = True
KERNEL_PROFILING = True

# Get SHD dataset
dataset = SHD(save_to='../data', train=TRAIN, transform=CropTime(max=1000 * 1000.0))

# Preprocess
spikes = []
labels = []
for i in range(3 * BATCH_SIZE):
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
network = Network()
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input, record_spikes=True)
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                        NUM_HIDDEN, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output)

    # Connections
    in_hid = Connection(input, hidden, Dense(np.load("in_hid.npy").reshape((num_input, NUM_HIDDEN))),
                        Exponential(5.0))
    #Connection(hidden, hidden, Dense(np.load("hid_hid.npy").reshape((NUM_HIDDEN, NUM_HIDDEN))),
    #           Exponential(5.0))
    hid_out= Connection(hidden, output, Dense(np.load("hid_out.npy").reshape((NUM_HIDDEN, num_output))),
                        Exponential(5.0))

max_example_timesteps = int(np.ceil(latest_spike_time / DT))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 reg_lambda_upper=2.5e-09, reg_lambda_lower=2.5e-09, 
                                 reg_nu_upper=14, max_spikes=1500, 
                                 optimiser=Adam(0.0), batch_size=BATCH_SIZE, 
                                 kernel_profiling=KERNEL_PROFILING)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar", Checkpoint(serialiser),
                     SpikeRecorder(input, key="in_spikes", example_filter=[0,32,64]),
                     SpikeRecorder(hidden, key="hidden_spikes", example_filter=[0,32,64]),
                     VarRecorder(output, "v", key="out_v", example_filter=[0,32,64]),
                     VarRecorder(hidden, "v", key="hidden_v", example_filter=[0,32,64]),
                     VarRecorder(hidden, genn_var="LambdaV", key="hidden_lambda_v", example_filter=[0,32,64]),
                     VarRecorder(hidden, genn_var="LambdaI", key="hidden_lambda_i", example_filter=[0,32,64]),
                     VarRecorder(hidden, genn_var="SpikeCount", key="hidden_spike_count", example_filter=[0,32,64]),
                     VarRecorder(hidden, genn_var="SpikeCountBackBatch", key="hidden_spike_count_back", example_filter=[0,32,64]),
                     VarRecorder(output, genn_var="LambdaV", key="out_lambda_v", example_filter=[0,32,64]),
                     VarRecorder(output, genn_var="LambdaI", key="out_lambda_i", example_filter=[0,32,64]),
                     ConnVarRecorder(in_hid, "g", key="in_hid_g", example_filter=[0,32,64]),
                     ConnVarRecorder(in_hid, "Gradient", key="in_hid_grad", example_filter=[0,32,64]),
                     ConnVarRecorder(hid_out, "g", key="hid_out_g", example_filter=[0,32,64]),
                     ConnVarRecorder(hid_out, "Gradient", key="hid_out_grad", example_filter=[0,32,64])]
        metrics, cb_data  = compiled_net.train({input: spikes},
                                               {output: labels},
                                               num_epochs=NUM_EPOCHS, shuffle=False,
                                               callbacks=callbacks)
        np.savez("in_spike_times_hack", *cb_data["in_spikes"][0])
        np.savez("in_spike_ids_hack", *cb_data["in_spikes"][1])
        np.savez("hidden_spike_times_hack", *cb_data["hidden_spikes"][0])
        np.savez("hidden_spike_ids_hack", *cb_data["hidden_spikes"][1])
        np.save("out_v_hack.npy", cb_data["out_v"])
        np.save("hidden_v_hack.npy", cb_data["hidden_v"])
        np.save("hidden_lambda_v_hack.npy", cb_data["hidden_lambda_v"])
        np.save("hidden_lambda_i_hack.npy", cb_data["hidden_lambda_i"])
        np.save("hidden_spike_count_hack.npy", cb_data["hidden_spike_count"])
        np.save("hidden_spike_count_back_hack.npy", cb_data["hidden_spike_count_back"])
        np.save("out_lambda_v_hack.npy", cb_data["out_lambda_v"])
        np.save("out_lambda_i_hack.npy", cb_data["out_lambda_i"])
        np.save("in_hid_g_hack.npy", cb_data["in_hid_g"])
        np.save("in_hid_grad_hack.npy", cb_data["in_hid_grad"])
        np.save("hid_out_g_hack.npy", cb_data["hid_out_g"])
        np.save("hid_out_grad_hack.npy", cb_data["hid_out_grad"])
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
