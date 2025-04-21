import matplotlib.pyplot as plt
import mnist
import numpy as np

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from time import perf_counter
from ml_genn.utils.data import linear_latency_encode_data

from ml_genn.compilers.event_prop_compiler import default_params

NUM_INPUT = 784
NUM_HIDDEN = 128
NUM_OUTPUT = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
EXAMPLE_TIME = 20.0
DT = 1.0
SPARSITY = 0.1
TRAIN = True
KERNEL_PROFILING = True
PLOT_REWIRING = True

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
spikes = linear_latency_encode_data(
    mnist.train_images() if TRAIN else mnist.test_images(),
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)

serialiser = Numpy("latency_mnist_checkpoints")
network = SequentialNetwork(default_params)
with network:
    # Populations
    input = InputLayer(SpikeInput(max_spikes=BATCH_SIZE * NUM_INPUT),
                                  NUM_INPUT)
    hidden = Layer(FixedProbability(SPARSITY, Normal(mean=0.078, sd=0.045)), 
                   LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0, tau_refrac=None),
                   NUM_HIDDEN, Exponential(5.0))
    output = Layer(Dense(Normal(mean=0.2, sd=0.37)),
                   LeakyIntegrate(tau_mem=20.0, readout="avg_var"),
                   NUM_OUTPUT, Exponential(5.0))

max_example_timesteps = int(np.ceil(EXAMPLE_TIME / DT))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 optimiser=Adam(1e-2), batch_size=BATCH_SIZE, dt=DT,
                                 deep_r_conns=[hidden],
                                 deep_r_l1_strength=0.00000001,
                                 deep_r_record_rewirings=({} if not PLOT_REWIRING 
                                                          else {"in_hid_rewiring": hidden}),
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
            print(f"Gradient batch reduce time = {compiled_net.genn_model.get_custom_update_time('GradientBatchReduce')}")
            print(f"Gradient learn time = {compiled_net.genn_model.get_custom_update_time('GradientLearn')}")
            print(f"Reset time = {compiled_net.genn_model.get_custom_update_time('Reset')}")
            print(f"Softmax1 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax1')}")
            print(f"Softmax2 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax2')}")
            print(f"Softmax3 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax3')}")
        
        # Loop through deep-r connections and plot rewiring curves
        if PLOT_REWIRING:
            fig, axis = plt.subplots()
            transpose = list(zip(*cb_data["in_hid_rewiring"]))
            axis.plot(transpose[0], label=c.name)
            axis.legend()
            axis.set_ylabel("Num rewirings")
            axis.set_xlabel("Batch")
            plt.show()

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
