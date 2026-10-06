import numpy as np
import torch

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import VarRecorder
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import AvgPoolConv2D, AvgPoolDense2D, Conv2D, Dense
from ml_genn.neurons import LeakyIntegrateFire, Neuron, SpikeInput
from ml_genn.utils.model import NeuronModel
from ml_genn.utils.snippet import ConstantValueDescriptor
from time import perf_counter
from tonic.datasets import DVSGesture
from tonic.transforms import Compose, CropTime, Denoise, Downsample

from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

BATCH_SIZE = 128

class SNNTorchLIF(Neuron):
    beta =  ConstantValueDescriptor()
    v_thresh =  ConstantValueDescriptor()

    def __init__(self, beta, v_thresh = 1.0, readout=None):
        super().__init__(readout)
        self.beta = float(beta)
        self.v_thresh = float(v_thresh)

    def get_model(self, population, dt: float, batch_size: int) -> NeuronModel:
        # Extract tau mem and R parameters that WOULD be exported to NIR
        tau_mem = dt / (1.0 - self.beta)
        r = tau_mem / dt
        
        r_factor = (dt / tau_mem) * (tau_mem / dt)
 
        v_scale = 1.0 / r_factor  # scaling factor from tau_mem+r circuit
        print(v_scale)
        model = {
            "params": [("Beta", "scalar"), ("VThresh", "scalar")],
            "vars": [("V", "scalar")],

            "sim_code":
                """
                const bool spike = (V >= VThresh);
                V = (Beta * V) + Isyn;
                """,
            "threshold_condition_code":
                """
                spike
                """,
            "reset_code":
                """
                V -= VThresh;
                """}
        return NeuronModel(model, "V",
                           {"Beta": self.beta, "VThresh": self.v_thresh},# * v_scale},
                           {"V": 0.0})


    
def reshape_conv_weight(weight):
    # PyTorch uses (out_channels, in_channels, kernel_height, kernel_width)
    # mlGeNN uses (kernel_height, kernel_width, in_channels, out_channels)
    
    return np.moveaxis(weight.numpy(), (0, 1, 2, 3), (3, 2, 0, 1))

def reshape_dense_weight(weight):
    # PyTorch uses (out_channels, in_channels)
    # mlGeNN uses (in_channels, out_channels)
    return np.transpose(weight.numpy())

def reshape_post_pool_weight(weight, in_channels, in_size):
    # PyTorch uses (out_channels, in_channels, in_height, in_width)
    # mlGeNN uses (in_height, in_width, in_channels, out_channels)
    # Unflatten weight
    weight = np.reshape(weight.numpy(), (weight.shape[0], in_channels, in_size, in_size))

    # Re-order axes into GeNN/TF
    weight = np.moveaxis(weight, (0, 1, 2, 3), (3, 2, 0, 1))

    # Reflatten into linear weight
    return np.reshape(weight, (-1, weight.shape[-1]))

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
for events, label in dataset:
#for i in range(10):
#    events, label = dataset[i]
    spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                          sensor_size, dt=1.0,
                                          histogram_thresh=1))
    labels.append(label)

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}, Num input {num_input}, Num output {num_output}")

# Load pytorch checkpoint
checkpoint = torch.load("PTQ_time_window-1ms-snntorch_dvsgesture_model.pth",
                        map_location=torch.device("cpu"), weights_only=True)

# Create sequential model
network = SequentialNetwork()
with network:
    input = InputLayer(SpikeInput(max_spikes=BATCH_SIZE * max_spikes), sensor_size)
    hidden1 = Layer(Conv2D(weight=reshape_conv_weight(checkpoint["0.weight"]), 
                           filters=16, conv_size=5, conv_strides=2, conv_padding=1),
                    SNNTorchLIF(beta=checkpoint["1.beta"], v_thresh=checkpoint["1.threshold"]))
    hidden2 = Layer(Conv2D(weight=reshape_conv_weight(checkpoint["2.weight"]), 
                           filters=16, conv_size=3, conv_padding="same"),
                    SNNTorchLIF(beta=checkpoint["3.beta"], v_thresh=checkpoint["3.threshold"]))
    hidden3 = Layer(AvgPoolConv2D(weight=reshape_conv_weight(checkpoint["5.weight"]), 
                                  filters=8, conv_size=3, pool_size=2, conv_padding="same", sum=True),
                    SNNTorchLIF(beta=checkpoint["6.beta"], v_thresh=checkpoint["6.threshold"]))
    hidden4 = Layer(AvgPoolDense2D(weight=reshape_post_pool_weight(checkpoint["9.weight"], 8, 3), 
                                   pool_size=2, sum=True),
                    SNNTorchLIF(beta=checkpoint["10.beta"], v_thresh=checkpoint["10.threshold"]), 256)
    output = Layer(Dense(weight=reshape_dense_weight(checkpoint["11.weight"])), 
                   SNNTorchLIF(beta=checkpoint["12.beta"], v_thresh=checkpoint["12.threshold"], readout="spike_count"))

compiler = InferenceCompiler(dt=1.0, batch_size=BATCH_SIZE,
                             evaluate_timesteps=1000,
                             reset_in_syn_between_batches=True)
compiled_net = compiler.compile(network)

with compiled_net:
    #compiled_net.connection_populations[hidden1.connection()].pull_connectivity_from_device()
    #compiled_net.connection_populations[hidden2.connection()].pull_connectivity_from_device()
    #compiled_net.connection_populations[hidden3.connection()].pull_connectivity_from_device()
    
    #np.save("hidden1.npy", np.vstack((compiled_net.connection_populations[hidden1.connection()].get_sparse_pre_inds(),
    #                                  compiled_net.connection_populations[hidden1.connection()].get_sparse_post_inds())))
    #np.save("hidden2.npy", np.vstack((compiled_net.connection_populations[hidden2.connection()].get_sparse_pre_inds(),
    #                                  compiled_net.connection_populations[hidden2.connection()].get_sparse_post_inds())))
    #np.save("hidden3.npy", np.vstack((compiled_net.connection_populations[hidden3.connection()].get_sparse_pre_inds(),
    #                                  compiled_net.connection_populations[hidden3.connection()].get_sparse_post_inds())))
    
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    metrics, _ = compiled_net.evaluate({input: spikes}, {output: labels})
    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
    print(f"Time = {end_time - start_time}s")
