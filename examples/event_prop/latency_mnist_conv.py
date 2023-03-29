import numpy as np
import matplotlib.pyplot as plt
import mnist

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Conv2D, Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                linear_latency_encode_data)

from ml_genn.compilers.event_prop_compiler import default_params

NUM_INPUT = 784
NUM_HIDDEN = 128
NUM_OUTPUT = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
EXAMPLE_TIME = 20.0
DT = 1.0
SPARSITY = 1.0
TRAIN = True
KERNEL_PROFILING = True

labels = mnist.train_labels() if TRAIN else mnist.test_labels()
spikes = linear_latency_encode_data(
    mnist.train_images() if TRAIN else mnist.test_images(),
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)

serialiser = Numpy("latency_mnist_checkpoints")
network = SequentialNetwork(default_params)
with network:
    # Populations
    input = InputLayer(SpikeInput(max_spikes=BATCH_SIZE * calc_max_spikes(spikes)),
                                  (28, 28, 1), name="input")
    initial_hidden1_weight = Normal(mean=0.078, sd=0.045)
    hidden1 = Layer(Conv2D(initial_hidden1_weight, 16, 3, True),
                    LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                       tau_refrac=Nonee),
                  synapse=Exponential(5.0), name="hidden1")
    initial_hidden2_weight = Normal(mean=0.078, sd=0.045)
    connectivity2 = (Dense(initial_hidden2_weight) if SPARSITY == 1.0 
                     else FixedProbability(SPARSITY, initial_hidden2_weight))
    hidden2 = Layer(connectivity2, LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                                      tau_refrac=None),
                    NUM_HIDDEN, Exponential(5.0), name="hidden2")
    output = Layer(Dense(Normal(mean=0.2, sd=0.37)),
                   LeakyIntegrate(tau_mem=20.0, readout="avg_var"),
                   NUM_OUTPUT, Exponential(5.0), name="output")

max_example_timesteps = int(np.ceil(EXAMPLE_TIME / DT))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 optimiser=Adam(1e-2), batch_size=BATCH_SIZE,
                                 kernel_profiling=KERNEL_PROFILING)
    compiled_net = compiler.compile(network)

    with compiled_net:
        visualise_examples = [0, 32, 64, 96]
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        """
        callbacks = ["batch_progress_bar", Checkpoint(serialiser), 
                     SpikeRecorder(input, "in_spikes", visualise_examples),
                     SpikeRecorder(hidden, "hid_spikes", visualise_examples, record_spike_events=True),
                     SpikeRecorder(input, "in_spike_events", visualise_examples, record_spike_events=True),
                     SpikeRecorder(hidden, "hid_spike_events", visualise_examples),
                     VarRecorder(hidden, None, "hid_lambda_v", visualise_examples,
                                 genn_var="LambdaV"),
                     VarRecorder(output, None, "out_lambda_v", visualise_examples,
                                 genn_var="LambdaV"),
                     VarRecorder(output, "v", "out_v", visualise_examples),
                     VarRecorder(output, None, "out_sum_v", visualise_examples,
                                 genn_var="SumV"),
                     VarRecorder(output, None, "out_softmax", visualise_examples,
                                 genn_var="Softmax")]
        """
        metrics, cb_data  = compiled_net.train({input: spikes},
                                               {output: labels},
                                               num_epochs=NUM_EPOCHS, shuffle=True,
                                               callbacks=callbacks)
        compiled_net.save_connectivity((NUM_EPOCHS - 1,), serialiser)
        """
        fig, axes = plt.subplots(9, len(visualise_examples), sharex="col", sharey="row")
        axes[0, 0].set_ylabel("Input spikes")
        axes[1, 0].set_ylabel("Input spike events")
        axes[2, 0].set_ylabel("Hidden spikes")
        axes[3, 0].set_ylabel("Hidden spike events")
        
        axes[4, 0].set_ylabel("Hidden lambda V")
        axes[5, 0].set_ylabel("Output lambda V")
        axes[6, 0].set_ylabel("Output V")
        axes[7, 0].set_ylabel("Output sum V")
        axes[8, 0].set_ylabel("Output softmax")
        
        for j, e in enumerate(visualise_examples):
            axes[0, j].set_title(f"Example {e}")
            axes[0, j].scatter(cb_data["in_spikes"][0][j], cb_data["in_spikes"][1][j], s=2)
            axes[1, j].scatter(cb_data["in_spike_events"][0][j], cb_data["in_spike_events"][1][j], s=2)
            axes[2, j].scatter(cb_data["hid_spike_events"][0][j], cb_data["hid_spike_events"][1][j], s=2)
            axes[3, j].scatter(cb_data["hid_spikes"][0][j], cb_data["hid_spikes"][1][j], s=2)

            axes[4, j].plot(cb_data["hid_lambda_v"][j])
            axes[5, j].plot(cb_data["out_lambda_v"][j])
            axes[6, j].plot(cb_data["out_v"][j])
            axes[7, j].plot(cb_data["out_sum_v"][j])
            axes[8, j].plot(cb_data["out_softmax"][j])

            axes[8, j].set_xlabel("Time [ms]")
        plt.show()
        """
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
