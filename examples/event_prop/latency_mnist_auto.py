import numpy as np
import matplotlib.pyplot as plt
import mnist

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import AutoNeuron, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import AutoSyn
from ml_genn.callbacks import VarRecorder, SpikeRecorder, ConnVarRecorder

from time import perf_counter
from ml_genn.utils.data import linear_latency_encode_data

from ml_genn.compilers.event_prop_compiler import default_params

NUM_INPUT = 784
NUM_HIDDEN = 128
NUM_OUTPUT = 10
BATCH_SIZE = 32
NUM_EPOCHS = 20
EXAMPLE_TIME = 20.0
DT = 1.0
SPARSITY = 1.0
TRAIN = True
KERNEL_PROFILING = True
DEBUG = True
PLOTTING = False
RECORDING = False

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
spikes = linear_latency_encode_data(
    mnist.train_images() if TRAIN else mnist.test_images(),
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)

print(spikes[0])
serialiser = Numpy("latency_mnist_checkpoints")
network = SequentialNetwork(default_params)
with network:
    # Populations
    input = InputLayer(SpikeInput(max_spikes=BATCH_SIZE * NUM_INPUT),
                                  NUM_INPUT,record_spikes=True)
    initial_hidden_weight = Normal(mean=0.078, sd=0.045)
    connectivity = (Dense(initial_hidden_weight) if SPARSITY == 1.0 
                    else FixedProbability(SPARSITY, initial_hidden_weight))
    hidden = Layer(connectivity, AutoNeuron([("V","scalar",0.0)], [("taum","scalar",20.0), ("theta","scalar",1.0)], {"V": "(-V+I)/taum"}, "V-theta", {"V": "0"}, solver="linear_euler"),
                   NUM_HIDDEN, AutoSyn(vars=[("I","scalar",0.0)],params=[("taus","scalar",5.0)],ode={"I": "-I/taus"},jumps={"I": "I+w"},w_name="w",inject_current="I",solver="linear_euler"),record_spikes=True)
    output = Layer(Dense(Normal(mean=0.2, sd=0.37)),
                   AutoNeuron([("V","scalar",0.0)], [("TauM","scalar",20.0), ("theta","scalar",1.0)], {"V": "(-V+I)/TauM"}, "", {"V": "0"}, solver="linear_euler", readout="avg_var"),
                   NUM_OUTPUT, AutoSyn(vars=[("I","scalar",0.0)],params=[("taus","scalar",5.0)],ode={"I": "-I/taus"},jumps={"I": "I+w"},w_name="w",inject_current="I",solver="linear_euler"))

max_example_timesteps = int(np.ceil(EXAMPLE_TIME / DT))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 optimiser=Adam(1e-2), batch_size=BATCH_SIZE,
                                 kernel_profiling=KERNEL_PROFILING,dt=DT)
    compiled_net = compiler.compile(network)

    with compiled_net:
        visualise_examples = [0, 32, 64, 96]
        # Evaluate model on numpy dataset
        start_time = perf_counter()

        if RECORDING:
            callbacks = [
                         SpikeRecorder(input, key="spikes_input",example_filter=[ 32, 33 ]),
                         SpikeRecorder(hidden, key="spikes_hidden",example_filter=[ 32, 33]),
                         #VarRecorder(hidden,genn_var="V",key="Vhid",neuron_filter=[0, 1]),
                         #VarRecorder(output,genn_var="V",key="Vout"),
                         VarRecorder(hidden,genn_var="LambdaV",key="LVhid",neuron_filter=[0, 1],example_filter=[ 32, 33 ]),
                         VarRecorder(output,genn_var="LambdaV",key="LVout",example_filter=[ 32, 33 ]),
                         #ConnVarRecorder(hidden.connection(), "w", "whidout",example_filter=[32]),
            ]
        else:
            callbacks = ["batch_progress_bar", Checkpoint(serialiser),]
        
        metrics, cb_data  = compiled_net.train({input: spikes},
                                               {output: labels},
                                               num_epochs=NUM_EPOCHS, shuffle=False,
                                               callbacks=callbacks)

        print(metrics.items())
        #print(cb_data)
        #print(type(cb_data))
        if PLOTTING:
            LVout = np.asarray(cb_data["LVout"])
            LVhid = np.asarray(cb_data["LVhid"])
            sin = cb_data["spikes_input"]
            print(sin[0][0])
            shid =  cb_data["spikes_hidden"]
            plt.figure()
            for i in range(20):
                plt.plot(np.arange(20/DT)+i*(20/DT+1),LVhid[i,:,:])
            plt.figure()
            for i in range(20):
                plt.plot(np.arange(20/DT)+i*(20/DT+1),LVout[i,:,:])
            plt.figure()
            plt.scatter(sin[0][0],sin[1][0],s=1)
            plt.figure()
            print(len(shid[0]))
            for i in range(20):
                plt.scatter(np.asarray(shid[0][i])+i*(20/DT+1),shid[1][i],s=1)
            plt.show()

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
