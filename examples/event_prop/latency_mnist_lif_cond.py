import logging
import numpy as np
import mnist

from ml_genn import Connection, Population, Network
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, UserNeuron, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from time import perf_counter
from ml_genn.utils.data import linear_latency_encode_data

NUM_INPUT = 784
NUM_HIDDEN = 128
NUM_OUTPUT = 10
BATCH_SIZE = 32
NUM_EPOCHS = 20
EXAMPLE_TIME = 30.0
DT = 1.0
SPARSITY = 0.1
TRAIN = True
KERNEL_PROFILING = True

lif_cond_neuron = UserNeuron(vars={"v": ("(((c_mem / tau_mem) * -v) + (Isyn * -v) + (GI * (i_rev - v)))/ c_mem", "0.0")},
                             threshold="v - 15.0",
                             additional_input_vars=["GI"],
                             output_var_name="v",
                             param_vals={"c_mem": 1.0, "tau_mem": 20.0, "i_rev": -5.0},
                             var_vals={"v": 0.0})

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
spikes = linear_latency_encode_data(
    mnist.train_images() if TRAIN else mnist.test_images(),
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)

logging.basicConfig(level=logging.DEBUG)
serialiser = Numpy("latency_mnist_checkpoints")
network = Network()
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * NUM_INPUT),
                       NUM_INPUT)
    hidden = Population(lif_cond_neuron, NUM_HIDDEN)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var"),
                        NUM_OUTPUT)

    # Connections
    input_hidden_exc = Connection(input, hidden, FixedProbability(SPARSITY, Normal(mean=0.0, sd=0.8)),
                                  Exponential(5.0))
    input_hidden_inh = Connection(input, hidden, FixedProbability(SPARSITY, Normal(mean=0.0, sd=0.8)),
                                  Exponential(5.0), target_input_var="GI")

    Connection(hidden, output, Dense(Normal(mean=0.2, sd=0.37)),
               Exponential(5.0))

max_example_timesteps = int(np.ceil(EXAMPLE_TIME / DT))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 batch_size=BATCH_SIZE, dt=DT,
                                 kernel_profiling=KERNEL_PROFILING)

    cond_weight_range = (0.0, np.finfo(np.float32).max)
    compiled_net = compiler.compile(network, optimisers={"all_connections": {"weight": "adam"},
                                                         input_hidden_exc: {"weight": Adam(clamp_var=cond_weight_range)},
                                                         input_hidden_inh: {"weight": Adam(clamp_var=cond_weight_range)}})


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
                                 batch_size=BATCH_SIZE, dt=DT)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels})
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
