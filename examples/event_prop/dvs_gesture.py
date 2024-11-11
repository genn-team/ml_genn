import numpy as np
import mnist

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import (AvgPool2D, AvgPoolConv2D, AvgPoolDense2D, 
                                  Conv2D, Dense, FixedProbability)
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic.datasets import DVSGesture
from tonic.transforms import Downsample

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)
from pygenn import PlogSeverity
from ml_genn.compilers.event_prop_compiler import default_params

BATCH_SIZE = 32
NUM_EPOCHS = 50
DT = 1.0
TRAIN = True
KERNEL_PROFILING = True
NUM_DENSE = 64
NUM_FILTERS = 64
NUM_CONV = 5

# Load dataset
dataset = DVSGesture(save_to="../data", train=TRAIN,
                     transform=Downsample(spatial_factor=0.25))
num_output = len(dataset.classes)

# Preprocess
spikes = []
labels = []
for i in range(BATCH_SIZE):#len(dataset)):
    events, label = dataset[i]
    spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                          (32, 32, 2), dt=DT,
                                          histogram_thresh=1))
    labels.append(label)


max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

serialiser = Numpy("dvs_gesture_checkpoints")
network = SequentialNetwork(default_params)
with network:
    input = InputLayer(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                                  (32, 32, 2))
    
    conv_sd = np.sqrt(2 / (3 * 3 * NUM_FILTERS))
    conv_weight = Normal(sd=conv_sd)
    Layer(Conv2D(conv_weight, filters=NUM_FILTERS, conv_size=(3, 3), 
                 conv_padding="same"),
          LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0, tau_refrac=None),
          synapse=Exponential(5.0))
    
    for i in range(NUM_CONV - 1):
        # **HACK** issue 119
        Layer(AvgPoolConv2D(np.random.normal(scale=conv_sd, size=(3, 3, NUM_FILTERS, NUM_FILTERS)), 
                            filters=NUM_FILTERS, conv_size=(3, 3), 
                            conv_padding="same", pool_size=2),
              LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0, tau_refrac=None),
              synapse=Exponential(5.0))
    
    # **NOTE** original architecture used AvgPoolDense2D but these aren't currently learnable
    # **THINK** plain avgpool is a bit dubious in an SNN as weights of 1 aren't necessarily ever meaningful
    #Layer(AvgPool2D(pool_size=2, flatten=True),
    #      LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0, tau_refrac=None),
    #      synapse=Exponential(5.0))
    
    dense_sd = (1.0 / np.sqrt(NUM_FILTERS))
    output = Layer(Dense(Normal(sd=dense_sd)),
                   LeakyIntegrate(tau_mem=20.0, readout="avg_var"),
                   num_output, Exponential(5.0))

max_example_timesteps = int(np.ceil(latest_spike_time / DT))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 #reg_lambda_upper=4e-09, reg_lambda_lower=4e-09,  reg_nu_upper=14, 
                                 max_spikes=500, pop_max_spikes={input: 6000},
                                 strict_buffer_checking=True,
                                 optimiser=Adam(0.001), batch_size=BATCH_SIZE, 
                                 kernel_profiling=KERNEL_PROFILING)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        metrics, _  = compiled_net.train({input: spikes},
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
