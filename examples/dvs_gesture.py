import numpy as np
import torch

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import BatchProgressBar, SpikeRecorder
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
RECORD = False

class SNNTorchLIF(Neuron):
    beta =  ConstantValueDescriptor()
    v_thresh =  ConstantValueDescriptor()

    def __init__(self, beta, v_thresh = 1.0, readout=None):
        super().__init__(readout)
        self.beta = float(beta)
        self.v_thresh = float(v_thresh)

    def get_model(self, population, dt: float, batch_size: int) -> NeuronModel:
        model = {
            "params": [("Beta", "scalar"), ("VThresh", "scalar")],
            "vars": [("V", "scalar")],

            "sim_code":
                """
                V = (Beta * V) + Isyn;
                """,
            "threshold_condition_code":
                """
                V >= VThresh
                """,
            "reset_code":
                """
                V -= VThresh;
                """}
        return NeuronModel(model, "V",
                           {"Beta": self.beta, "VThresh": self.v_thresh},
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
                                        Downsample(spatial_factor=0.25)]))
sensor_size = (32, 32, 2)

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = int(np.prod(sensor_size))
num_output = len(dataset.classes)

# Preprocess dataset
spikes = []
labels = []
for events, label in dataset:
#for i in range(1):
    #events, label = dataset[i]
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
                    SNNTorchLIF(beta=checkpoint["1.beta"], v_thresh=checkpoint["1.threshold"]),
                    record_spikes=RECORD)
    hidden2 = Layer(Conv2D(weight=reshape_conv_weight(checkpoint["2.weight"]), 
                           filters=16, conv_size=3, conv_padding="same"),
                    SNNTorchLIF(beta=checkpoint["3.beta"], v_thresh=checkpoint["3.threshold"]),
                    record_spikes=RECORD)
    hidden3 = Layer(AvgPoolConv2D(weight=reshape_conv_weight(checkpoint["5.weight"]), 
                                  filters=8, conv_size=3, pool_size=2, conv_padding="same", sum=True),
                    SNNTorchLIF(beta=checkpoint["6.beta"], v_thresh=checkpoint["6.threshold"]),
                    record_spikes=RECORD)
    hidden4 = Layer(AvgPoolDense2D(weight=reshape_post_pool_weight(checkpoint["9.weight"], 8, 3), 
                                   pool_size=2, sum=True),
                    SNNTorchLIF(beta=checkpoint["10.beta"], v_thresh=checkpoint["10.threshold"]), 
                    256, record_spikes=RECORD)
    output = Layer(Dense(weight=reshape_dense_weight(checkpoint["11.weight"])), 
                   SNNTorchLIF(beta=checkpoint["12.beta"], v_thresh=checkpoint["12.threshold"], readout="spike_count"),
                   record_spikes=RECORD)

max_example_timesteps = int(np.ceil(latest_spike_time / 1.0))
compiler = InferenceCompiler(dt=1.0, batch_size=BATCH_SIZE,
                             evaluate_timesteps=max_example_timesteps,
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

    if RECORD:
        callbacks = [BatchProgressBar(),
                     SpikeRecorder(hidden1, key="hidden1_spikes"),
                     SpikeRecorder(hidden2, key="hidden2_spikes"),
                     SpikeRecorder(hidden3, key="hidden3_spikes"),
                     SpikeRecorder(hidden4, key="hidden4_spikes"),
                     SpikeRecorder(output, key="output_spikes")]
    else:
        callbacks = [BatchProgressBar()]

    # Evaluate model on numpy dataset
    start_time = perf_counter()
    metrics, cb_data = compiled_net.evaluate({input: spikes}, {output: labels},
                                             callbacks=callbacks )
    if RECORD:
        np.savez(f"hidden1_spikes", times=cb_data["hidden1_spikes"][0][0], ids=cb_data["hidden1_spikes"][1][0])
        np.savez(f"hidden2_spikes", times=cb_data["hidden2_spikes"][0][0], ids=cb_data["hidden2_spikes"][1][0])
        np.savez(f"hidden3_spikes", times=cb_data["hidden3_spikes"][0][0], ids=cb_data["hidden3_spikes"][1][0])
        np.savez(f"hidden4_spikes", times=cb_data["hidden4_spikes"][0][0], ids=cb_data["hidden4_spikes"][1][0])
        np.savez(f"output_spikes", times=cb_data["output_spikes"][0][0], ids=cb_data["output_spikes"][1][0])

    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
    print(f"Time = {end_time - start_time}s")
