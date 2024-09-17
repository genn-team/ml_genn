import numpy as np

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import Checkpoint, OptimiserParamSchedule, SpikeRecorder, VarRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from time import perf_counter
from ml_genn.utils.data import generate_yin_yang_dataset

from ml_genn.compilers.event_prop_compiler import default_params

import matplotlib.pyplot as plt

NUM_INPUT = 4
NUM_HIDDEN = 100
NUM_OUTPUT = 3
BATCH_SIZE = 512
NUM_EPOCHS = 10
NUM_TRAIN = BATCH_SIZE * 10 * NUM_OUTPUT
NUM_TEST = BATCH_SIZE  * 2 * NUM_OUTPUT
EXAMPLE_TIME = 30.0
DT = 0.01
TRAIN = True
KERNEL_PROFILING = True

spikes, labels = generate_yin_yang_dataset(NUM_TRAIN if TRAIN else NUM_TEST, 
                                           EXAMPLE_TIME - (4 * DT))

# Plot training data
fig, axis = plt.subplots()
axis.scatter([d.spike_times[0]  for d in spikes], [d.spike_times[1] for d in spikes], c=labels)

serialiser = Numpy("yin_yang_checkpoints")
network = SequentialNetwork(default_params)
with network:
    # Populations
    input = InputLayer(SpikeInput(max_spikes=BATCH_SIZE * NUM_INPUT),
                                  NUM_INPUT, record_spikes=True)
    hidden = Layer(Dense(Normal(mean=1.5, sd=0.78)), 
                   LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                      tau_refrac=None),
                   NUM_HIDDEN, Exponential(5.0), record_spikes=True)
    output = Layer(Dense(Normal(mean=0.93, sd=0.1)),
                   LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                      tau_refrac=None,
                                      readout="first_spike_time"),
                   NUM_OUTPUT, Exponential(5.0), record_spikes=True)

max_example_timesteps = int(np.ceil(EXAMPLE_TIME / DT))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 optimiser=Adam(0.003, 0.9, 0.99), batch_size=BATCH_SIZE,
                                 softmax_temperature=0.5, ttfs_alpha=0.1, dt=DT,
                                 kernel_profiling=KERNEL_PROFILING)
    compiled_net = compiler.compile(network)

    with compiled_net:
        def alpha_schedule(epoch, alpha):
            return 0.003 * (0.998 ** epoch)

        # Evaluate model on dataset
        start_time = perf_counter()
        examples = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        examples_back = [512, 522, 532, 542, 552, 562, 572, 582, 592, 602]
        callbacks = ["batch_progress_bar", Checkpoint(serialiser), 
                     OptimiserParamSchedule("alpha", alpha_schedule),
                     SpikeRecorder(input, key="InputSpikes", example_filter=examples),
                     SpikeRecorder(hidden, key="HiddenSpikes", example_filter=examples),
                     SpikeRecorder(output, key="OutputSpikes", example_filter=examples),
                     VarRecorder(output, key="OutputTTFS", genn_var="TFirstSpike", example_filter=examples),
                     VarRecorder(output, key="OutputLambdaV", genn_var="LambdaV", example_filter=examples_back)]
        metrics, cb_data  = compiled_net.train({input: spikes},
                                               {output: labels},
                                               num_epochs=NUM_EPOCHS, shuffle=False,
                                               callbacks=callbacks)
        for e in range(NUM_EPOCHS):
            fig, axes = plt.subplots(4, 10, sharex="col", sharey="row")
            timesteps = np.arange(0.0, EXAMPLE_TIME, DT)
            for i in range(10):
                #in_spikes = [
                axes[0,i].set_title(f"Example {(e * 10) + i}")
                axes[0,i].scatter(cb_data["InputSpikes"][0][(e * 10) + i], cb_data["InputSpikes"][1][(e * 10) + i], s=1)
                axes[1,i].scatter(cb_data["HiddenSpikes"][0][(e * 10) + i], cb_data["HiddenSpikes"][1][(e * 10) + i], s=1)
                axes[2,i].scatter(cb_data["OutputSpikes"][0][(e * 10) + i], cb_data["OutputSpikes"][1][(e * 10) + i], s=1)
                
                axes[2,i].scatter(-cb_data["OutputTTFS"][(e * 10) + i][-1,:], np.arange(3), marker="X", alpha=0.5)
                
                #for i in NUM_OUTPUT:
                for j in range(NUM_OUTPUT):
                    axes[3,i].plot(timesteps, (j * 0.002) + cb_data["OutputLambdaV"][((e * 10) + i)][::-1,j],
                                linestyle=("-" if labels[examples[i]] == j else "--"))
                    #axes[3,i].plot(j + cb_data["OutputLambdaI"][BATCH_SIZE + (i * 10)][::-1,j])
                axes[3,i].set_xlabel("Time [ms]")
                axes[3,i].set_xlim((0, EXAMPLE_TIME))
        
            axes[0,0].set_ylabel("Input neuron ID")
            axes[1,0].set_ylabel("Hidden neuron ID")
            axes[2,0].set_ylabel("Output neuron ID")
        plt.show()

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
