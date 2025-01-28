import numpy as np
import mnist

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import FluctuationDrivenExponential, Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from time import perf_counter
from ml_genn.utils.data import linear_latency_encode_data

from ml_genn.compilers.event_prop_compiler import default_params

NUM_INPUT = 784
NUM_HIDDEN_E = 100
NUM_HIDDEN_I = 25
NUM_OUTPUT = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
EXAMPLE_TIME = 20.0
DT = 1.0
TRAIN = True
KERNEL_PROFILING = True
RATE = 1.0# / (EXAMPLE_TIME * 1E-3)

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
spikes = linear_latency_encode_data(
    mnist.train_images() if TRAIN else mnist.test_images(),
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)

serialiser = Numpy("latency_mnist_dalian_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * NUM_INPUT),
                       NUM_INPUT)
    hidden_e = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0, tau_refrac=None),
                          NUM_HIDDEN_E)
    hidden_i = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0, tau_refrac=None),
                          NUM_HIDDEN_I)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var"), NUM_OUTPUT)
    
    # Connections
    fluc_driven_weight = FluctuationDrivenExponential(n_exc_ff=NUM_INPUT,
                                                      n_exc_rec=0,
                                                      n_inh=NUM_HIDDEN_I,
                                                      nu=RATE)
    Connection(input, hidden_e,
               Dense(fluc_driven_weight), 
               Exponential(5.0),
               polarity="excitatory")
    Connection(input, hidden_i,
               Dense(fluc_driven_weight), 
               Exponential(5.0),
               polarity="excitatory")
    Connection(hidden_i, hidden_i,
               Dense(fluc_driven_weight), 
               Exponential(5.0),
               polarity="inhibitory")
    Connection(hidden_i, hidden_e,
               Dense(fluc_driven_weight), 
               Exponential(5.0),
               polarity="inhibitory")
    Connection(hidden_e, output,
               Dense(Normal(mean=0.2, sd=0.37)),
               Exponential(5.0))

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
