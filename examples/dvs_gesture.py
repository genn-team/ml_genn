import numpy as np
import torch

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import LeakyIntegrateFire, SpikeInput
from ml_genn.connectivity import AvgPoolConv2D, AvgPoolDense2D, Conv2D, Dense
from ml_genn.callbacks import VarRecorder
from time import perf_counter
from tonic.datasets import DVSGesture
from tonic.transforms import Compose, CropTime, Denoise, Downsample

from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

BATCH_SIZE = 128

def reshape_conv_weight(weight):
    # PyTorch uses (out_channels, in_channels​, kernel_height, kernel_width)
    # mlGeNN uses (kernel_height, kernel_width, in_channels, out_channels)
    
    return np.moveaxis(weight.cpu().numpy(), (0, 1, 2, 3), (3, 2, 0, 1))

def reshape_dense_weight(weight):
    return np.transpose(weight.cpu().numpy())

# Load DVS gesture, cropping time and downsampling
dataset = DVSGesture(save_to="./data", train=False, 
                     transform=Compose([Denoise(filter_time=10000),
                                        Downsample(spatial_factor=0.25),
                                        CropTime(max=1000 * 1000)]))
sensor_size = (32, 32, 2)

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = int(np.prod(sensor_size))
num_output = len(dataset.classes)

# Preprocess dataset
spikes = []
labels = []
#for events, label in dataset:
for i in range(10):
    events, label = dataset[i]
    spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                          sensor_size, dt=1.0,
                                          histogram_thresh=1))
    labels.append(label)

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}, Num input {num_input}, Num output {num_output}")

# Load pytorch checkpoint
checkpoint = torch.load("PTQ_time_window-1ms-snntorch_dvsgesture_model.pth")

# Create sequential model
network = SequentialNetwork()
with network:
    input = InputLayer(SpikeInput(max_spikes=BATCH_SIZE * max_spikes), sensor_size)
    Layer(Conv2D(weight=reshape_conv_weight(checkpoint["0.weight"]), filters=16, conv_size=5, conv_strides=2, conv_padding="same"),
          LeakyIntegrateFire(tau_mem=14.0))
    Layer(Conv2D(weight=reshape_conv_weight(checkpoint["2.weight"]), filters=16, conv_size=3, conv_padding="same"),
          LeakyIntegrateFire(tau_mem=14.0))
    Layer(AvgPoolConv2D(weight=reshape_conv_weight(checkpoint["5.weight"]), filters=8, conv_size=3, pool_size=2, conv_padding="same"),
          LeakyIntegrateFire(tau_mem=14.0))
    Layer(AvgPoolDense2D(weight=reshape_dense_weight(checkpoint["9.weight"]), pool_size=2),
          LeakyIntegrateFire(tau_mem=14.0), 256)
    output = Layer(Dense(weight=reshape_dense_weight(checkpoint["11.weight"])),
                   LeakyIntegrateFire(tau_mem=14.0, readout="spike_count"))

compiler = InferenceCompiler(dt=1.0, batch_size=BATCH_SIZE,
                             evaluate_timesteps=1000)
compiled_net = compiler.compile(network)

with compiled_net:
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    metrics, _ = compiled_net.evaluate({input: spikes}, {output: labels})
    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
    print(f"Time = {end_time - start_time}s")
