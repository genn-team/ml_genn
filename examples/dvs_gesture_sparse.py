import numpy as np

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import BatchProgressBar, SpikeRecorder
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import FixedProbability, Dense
from ml_genn.neurons import LeakyIntegrateFire, Neuron, SpikeInput
from ml_genn.utils.model import NeuronModel
from ml_genn.utils.snippet import ConstantValueDescriptor
from time import perf_counter
from tonic.datasets import DVSGesture
from tonic.transforms import Compose, CropTime, Denoise, Downsample

from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_spikes)

BATCH_SIZE = 16
RECORD = False
DT = 0.1
TAU_REF = 9

class SNNTorchLIF(Neuron):
    beta =  ConstantValueDescriptor()
    v_thresh =  ConstantValueDescriptor()

    def __init__(self, beta, v_thresh=1.0, tau_ref=0, readout=None):
        super().__init__(readout)
        self.beta = float(beta)
        self.v_thresh = float(v_thresh)
        self.tau_ref = int(tau_ref)

    def get_model(self, population, dt: float, batch_size: int) -> NeuronModel:
        model = {
            "params": [("Beta", "scalar"), ("VThresh", "scalar"), ("TauRef", "int")],
            "vars": [("V", "scalar"), ("TRef", "int")],

            "sim_code":
                """
                if (TRef > 0) {
                    TRef --;
                }
                V = (Beta * V) + Isyn;
                """,
            "threshold_condition_code":
                """
                ((V >= VThresh) && (TRef <= 0))
                """,
            "reset_code":
                """
                V -= VThresh;
                TRef = TauRef;
                """}
        return NeuronModel(model, "V",
                           {"Beta": self.beta, "VThresh": self.v_thresh, "TauRef": self.tau_ref},
                           {"V": 0.0, "TRef": 0.0})


def remap_width_height_channel(inds, width, height, num_channels):
    # Unravel indices into their y, x, c components
    ind_y, ind_x, ind_c = np.unravel_index(inds, (height, width, num_channels))
    
    # Ravel them back in x, y, c order
    return np.ravel_multi_index((ind_x, ind_y, ind_c), (width, height, num_channels))

def remap_channel_height_width(inds, width, height, num_channels):
    # Unravel indices into their y, x, c components
    ind_y, ind_x, ind_c = np.unravel_index(inds, (height, width, num_channels))
    
    # Ravel them back in c, y, x order
    return np.ravel_multi_index((ind_c, ind_y, ind_x), (num_channels, height, width))

def shuffle_channel_height_width(weights, pre_width, pre_height, pre_channels, num_post):
    # Reshape weights into (pre_height, pre_width, pre_channels, num_post) shape
    weights = np.reshape(weights, (pre_height, pre_width, pre_channels, num_post))
    
    # Re-order presynaptic axes
    weights = np.moveaxis(weights, (0, 1, 2, 3), (1, 2, 0, 3))
    
    # Reshape back into (num_pre, num_post) shape
    return np.reshape(weights, (pre_height * pre_width * pre_channels, num_post))

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
for i in range(5):
    events, label = dataset[i]
    
    spike_event_ids = (events["p"] +
                       (events["y"] * sensor_size[2]) + 
                       (events["x"] * sensor_size[1] * sensor_size[2]))
    spikes.append(preprocess_spikes(events["t"] / 1000.0, spike_event_ids,
                                    np.prod(sensor_size)))
    labels.append(label)

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}, Num input {num_input}, Num output {num_output}")

beta = 0.93**(DT / 1.0)

# Load weights
hidden1_conn = np.load("hidden1_qat.npy")
hidden2_conn = np.load("hidden2_qat.npy")
hidden3_conn = np.load("hidden3_qat.npy")
hidden4_conn = np.load("hidden4_qat.npy")
output_conn = np.reshape(np.load("output_qat.npy"), (256, 11))

# Input neurons need to be arranged in width, height, channel order
hidden1_conn[0] = remap_width_height_channel(hidden1_conn[0].astype(int), 32, 32, 2)

# Hidden 1 neurons need to be arranged in width, height, channel order
hidden1_conn[1] = remap_width_height_channel(hidden1_conn[1].astype(int), 15, 15, 16)
hidden2_conn[0] = remap_width_height_channel(hidden2_conn[0].astype(int), 15, 15, 16)

# Hidden 2 neurons need to be arranged in channel, height, width order
hidden2_conn[1] = remap_channel_height_width(hidden2_conn[1].astype(int), 15, 15, 16)
hidden3_conn[0] = remap_channel_height_width(hidden3_conn[0].astype(int), 15, 15, 16)

# Hidden 3 neurons need to be arranged in channel, height, width order
hidden3_conn[1] = remap_channel_height_width(hidden3_conn[1].astype(int), 7, 7, 8)
hidden4_conn = shuffle_channel_height_width(hidden4_conn, 7, 7, 8, 256)

# Create sequential model
network = SequentialNetwork()
with network:
    input = InputLayer(SpikeInput(max_spikes=BATCH_SIZE * max_spikes), sensor_size)
    hidden1 = Layer(FixedProbability(0.1, hidden1_conn[2]), 
                    SNNTorchLIF(beta=beta, v_thresh=1.0, tau_ref=TAU_REF),
                    15 * 15 * 16, record_spikes=RECORD)
    hidden2 = Layer(FixedProbability(0.1, hidden2_conn[2]), 
                    SNNTorchLIF(beta=beta, v_thresh=1.0, tau_ref=TAU_REF),
                    15 * 15 * 16, record_spikes=RECORD)
    hidden3 = Layer(FixedProbability(0.1, hidden3_conn[2]), 
                    SNNTorchLIF(beta=beta, v_thresh=1.0, tau_ref=TAU_REF),
                    7 * 7 * 8, record_spikes=RECORD)
    hidden4 = Layer(Dense(weight=hidden4_conn),
                    SNNTorchLIF(beta=beta, v_thresh=1.0, tau_ref=TAU_REF), 
                    256, record_spikes=RECORD)
    output = Layer(Dense(weight=output_conn), 
                   SNNTorchLIF(beta=beta, v_thresh=1.0, tau_ref=TAU_REF, readout="spike_count"),
                   11, record_spikes=RECORD)

# Copy in indices
hidden1.connection().connectivity.pre_ind = hidden1_conn[0].astype(int)
hidden1.connection().connectivity.post_ind = hidden1_conn[1].astype(int)
hidden2.connection().connectivity.pre_ind = hidden2_conn[0].astype(int)
hidden2.connection().connectivity.post_ind = hidden2_conn[1].astype(int)
hidden3.connection().connectivity.pre_ind = hidden3_conn[0].astype(int)
hidden3.connection().connectivity.post_ind = hidden3_conn[1].astype(int)


max_example_timesteps = int(np.ceil(latest_spike_time / DT))
print(max_example_timesteps)
compiler = InferenceCompiler(dt=DT, batch_size=BATCH_SIZE,
                             evaluate_timesteps=max_example_timesteps,
                             reset_in_syn_between_batches=True)
compiled_net = compiler.compile(network)

with compiled_net:
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
