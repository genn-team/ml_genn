import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import mnist

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense,FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                log_latency_encode_data)

from ml_genn.compilers.eprop_compiler import default_params

NUM_INPUT = 784
NUM_HIDDEN = 128
NUM_OUTPUT = 10
BATCH_SIZE = 128
NUM_EPOCHS = 2
SPARSITY = 0.1
TRAIN = True
KERNEL_PROFILING = False
PLOT_REWIRING = True

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
spikes = log_latency_encode_data(
    mnist.train_images() if TRAIN else mnist.test_images(),
    20.0, 51)

serialiser = Numpy("latency_mnist_checkpoints")
network = SequentialNetwork(default_params)
with network:
    # Populations
    input = InputLayer(SpikeInput(max_spikes=BATCH_SIZE * calc_max_spikes(spikes)),
                                  NUM_INPUT)
    hidden = Layer(FixedProbability(SPARSITY, Normal(sd=1.0 / np.sqrt(NUM_INPUT))), 
                   LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0,
                                      tau_refrac=5.0),
                   NUM_HIDDEN)
    output = Layer(Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))),
                   LeakyIntegrate(tau_mem=20.0, readout="sum_var"),
                   NUM_OUTPUT)

max_example_timesteps = int(np.ceil(calc_latest_spike_time(spikes)))
if TRAIN:
    compiler = EPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             optimiser="adam", batch_size=BATCH_SIZE,
                             #deep_r_conns=[hidden],
                             #deep_r_l1_strength=0.00000001,
                             #deep_r_record_rewirings=({} if not PLOT_REWIRING 
                             #                         else {hidden: "in_hid_rewiring"}),
                             kernel_profiling=KERNEL_PROFILING)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        metrics, cb_data  = compiled_net.train({input: spikes},
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
            print(f"Synapse dynamics time = {compiled_net.genn_model.synapse_dynamics_time}")
            print(f"Gradient batch reduce time = {compiled_net.genn_model.get_custom_update_time('GradientBatchReduce')}")
            print(f"Gradient learn time = {compiled_net.genn_model.get_custom_update_time('GradientLearn')}")
            print(f"Reset time = {compiled_net.genn_model.get_custom_update_time('Reset')}")
            print(f"Softmax1 time = {compiled_net.genn_model.get_custom_update_time('Softmax1')}")
            print(f"Softmax2 time = {compiled_net.genn_model.get_custom_update_time('Softmax2')}")
            print(f"Softmax3 time = {compiled_net.genn_model.get_custom_update_time('Softmax3')}")

        # Plot rewiring curves
        if PLOT_REWIRING:
            fig, axis = plt.subplots()
            transpose = list(zip(*cb_data["in_hid_rewiring"]))
            axis.plot(transpose[0])
            axis.set_ylabel("Num rewirings")
            axis.set_xlabel("Batch")
            plt.show()
else:
    # Load network state from final checkpoint
    network.load((NUM_EPOCHS - 1,), serialiser)

    compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
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
