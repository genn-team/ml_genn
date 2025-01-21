import matplotlib.pyplot as plt
import numpy as np
import mnist

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint, SpikeRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal, Wrapper
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from itertools import chain
from time import perf_counter
from ml_genn.utils.data import linear_latency_encode_data

from ml_genn.compilers.event_prop_compiler import default_params

IN_SP = 0.1
IN_EI_SPLIT = 0.5
NUM_INPUT = 784
NUM_HIDDEN = 128
NUM_OUTPUT = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
EXAMPLE_TIME = 20.0
DT = 1.0
TRAIN = True
KERNEL_PROFILING = True
PLOT_REWIRING = False

def get_exc_clipped_norm(sd):
    return Wrapper("NormalClipped", {"mean": 0.0, "sd": sd,
                                     "min": 0.0, "max": np.finfo(np.float32).max})

def get_inh_clipped_norm(sd):
    return Wrapper("NormalClipped", {"mean": 0.0, "sd": sd,
                                     "min": np.finfo(np.float32).min, "max": 0.0})


mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
spikes = linear_latency_encode_data(
    mnist.train_images() if TRAIN else mnist.test_images(),
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)

serialiser = Numpy("latency_mnist_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * NUM_INPUT),
                                  NUM_INPUT)
    
    # Hidden layer, split into E and I populations
    hidden_e = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                             tau_refrac=None),
                          NUM_HIDDEN // 2, record_spikes=True)
    
    hidden_i = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                             tau_refrac=None),
                          NUM_HIDDEN // 2, record_spikes=True)
    
    # Output population
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var"),
                        NUM_OUTPUT)
    
    # Because connections are always either excitatory or inhibitory, we have seperate E and I connections from input to hidden populations
    input_e_hidden_e = Connection(input, hidden_e, FixedProbability(IN_SP * IN_EI_SPLIT, 
                                                                    get_exc_clipped_norm(1.0), True),
                                  Exponential(5.0), name="Conn_InE_HidE")
    input_i_hidden_e = Connection(input, hidden_e, FixedProbability(IN_SP * (1.0 - IN_EI_SPLIT), 
                                                                    get_inh_clipped_norm(1.0), True),
                                  Exponential(5.0), name="Conn_InI_HidE")
    input_e_hidden_i = Connection(input, hidden_i, FixedProbability(IN_SP * IN_EI_SPLIT, 
                                                                    get_exc_clipped_norm(1.0), True),
                                  Exponential(5.0), name="Conn_InE_HidI")
    input_i_hidden_i = Connection(input, hidden_i, FixedProbability(IN_SP * (1.0 - IN_EI_SPLIT), 
                                                                    get_inh_clipped_norm(1.0), True),
                                  Exponential(5.0), name="Conn_InI_HidI")
    
    # Build lists of excitatory and inhibitory connections to apply deep-r to
    deep_r_exc_conns = [input_e_hidden_e, input_e_hidden_i]
    deep_r_inh_conns = [input_i_hidden_e, input_i_hidden_i]

    # Dense connectivity to output population
    Connection(hidden_e, output, Dense(Normal(mean=0.2, sd=0.37)), Exponential(5.0))
    Connection(hidden_i, output, Dense(Normal(mean=0.2, sd=0.37)), Exponential(5.0))                            

max_example_timesteps = int(np.ceil(EXAMPLE_TIME / DT))
if TRAIN:
    record_rewiring = {}
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 optimiser=Adam(1e-2), batch_size=BATCH_SIZE,
                                 deep_r_exc_conns=deep_r_exc_conns,
                                 deep_r_inh_conns=deep_r_inh_conns,
                                 deep_r_l1_strength=0.00000001,
                                 deep_r_record_rewirings=({} if not PLOT_REWIRING 
                                                          else {c: f"{c.name}_rewiring" 
                                                                for c in (deep_r_exc_conns + deep_r_inh_conns)}),
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
        
        # Loop through deep-r connections and plot rewiring curves
        if PLOT_REWIRING:
            fig, axis = plt.subplots()
            for c in (deep_r_exc_conns + deep_r_inh_conns):
                transpose = list(zip(*cb_data[f"{c.name}_rewiring"]))
                axis.plot(transpose[0], label=c.name)
            axis.legend()
            axis.set_ylabel("Num rewirings")
            axis.set_xlabel("Batch")
            plt.show()

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
