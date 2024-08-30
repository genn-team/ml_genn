import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, Callback
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense,FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic.datasets import SHD

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

from ml_genn.compilers.event_prop_compiler import default_params
import copy

NUM_HIDDEN = 256
N_DELAY = 10
T_DELAY = 30
BATCH_SIZE = 32
NUM_EPOCHS = 50
EXAMPLE_TIME = 20.0
AUGMENT_SHIFT = 40
P_BLEND = 0.5
DT = 1.0


class EaseInSchedule(Callback):
    def set_params(self, compiled_network, **kwargs):
        self._optimisers = [o for o, _ in compiled_network.optimisers]

    def on_batch_begin(self, batch):
        # Set parameter to return value of function
        for optimiser in self._optimisers:
            if optimiser.alpha < 0.001 :
                optimiser.alpha = (0.001 / 1000.0) * (1.05 ** batch)
            else:
                optimiser.alpha = 0.001


class Shift:
    def __init__(self, f_shift, sensor_size):
        self.f_shift = f_shift
        self.sensor_size = sensor_size

    def __call__(self, events: np.ndarray) -> np.ndarray:
        # Shift events
        events_copy = copy.deepcopy(events)
        events_copy["x"] = events_copy["x"] + np.random.randint(-self.f_shift, self.f_shift)

        # Delete out of bound events
        events_copy = np.delete(
            events_copy,
            np.where(
                (events_copy["x"] < 0) | (events_copy["x"] >= self.sensor_size[0])))
        return events_copy

class Blend:
    def __init__(self, p_blend, sensor_size):
        self.p_blend = p_blend
        self.n_blend = 7644
        self.sensor_size = sensor_size
    def __call__(self, dataset: list, classes: list) -> list:
        for i in range(self.n_blend):
            idx = np.random.randint(0,len(dataset))
            idx2 = np.random.randint(0,len(classes[dataset[idx][1]]))
            assert dataset[idx][1] == dataset[classes[dataset[idx][1]][idx2]][1]
            dataset.append((self.blend(dataset[idx][0], dataset[classes[dataset[idx][1]][idx2]][0]), dataset[idx][1]))

        return dataset

    def blend(self, X1, X2):
        X1 = copy.deepcopy(X1)
        X2 = copy.deepcopy(X2)
        mx1 = np.mean(X1["x"])
        mx2 = np.mean(X2["x"])
        mt1 = np.mean(X1["t"])
        mt2 = np.mean(X2["t"])
        X1["x"]+= int((mx2-mx1)/2)
        X2["x"]+= int((mx1-mx2)/2)
        X1["t"]+= int((mt2-mt1)/2)
        X2["t"]+= int((mt1-mt2)/2)
        X1 = np.delete(
            X1,
            np.where(
                (X1["x"] < 0) | (X1["x"] >= self.sensor_size[0]) | (X1["t"] < 0) | (X1["t"] >= 1000000)))
        X2 = np.delete(
            X2,
            np.where(
                (X2["x"] < 0) | (X2["x"] >= self.sensor_size[0]) | (X2["t"] < 0) | (X2["t"] >= 1000000)))
        mask1 = np.random.rand(X1["x"].shape[0]) < self.p_blend
        mask2 = np.random.rand(X2["x"].shape[0]) < self.p_blend
        X1_X2 = np.concatenate([X1[mask1], X2[mask2]])
        idx= np.argsort(X1_X2["t"])
        X1_X2 = X1_X2[idx]
        return X1_X2

class Delay:
    def __init__(self, t_delay, n_delay, num_input):
        self.n_delay = n_delay
        self.t_delay = t_delay * 1000
        self.num_input = num_input
    def __call__(self, events: np.ndarray) -> np.ndarray:
        delayed_events = [events]

        for n in range(1, self.n_delay):
            curr_event = copy.deepcopy(events)
            curr_event["x"] = curr_event["x"] + (n * self.num_input)
            curr_event["t"] = curr_event["t"] + (n * self.t_delay)
            curr_event = np.delete(curr_event, np.where(curr_event["t"] >= 1000000))
            delayed_events.append(curr_event)
        return np.concatenate(delayed_events)



# Get dataset
dataset = SHD(save_to="../data", train=True)

num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)
delay = Delay(T_DELAY, N_DELAY, num_input)
shift = Shift(AUGMENT_SHIFT, dataset.sensor_size)
blend = Blend(P_BLEND, dataset.sensor_size)

# Loop through dataset
max_spikes = 0
latest_spike_time = 0
raw_dataset = []
idx= np.arange(len(dataset),dtype= int)
np.random.shuffle(idx)
classes = [[] for _ in range(20)]
for i,j  in enumerate(idx[:7644]):
    events, label = dataset[j]
    events = np.delete(events, np.where(events["t"] >= 1000000))
    # Add raw events and label to list
    raw_dataset.append((events, label))
    classes[label].append(i)
    delay(events)
    # Calculate max spikes and max times
    max_spikes = max(max_spikes, len(events))
    latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)

spikes_val = []
labels_val = []
for i in idx[7644:]:
    events, label = dataset[i]

    events = np.delete(events, np.where(events["t"] >= 1000000))

    spikes_val.append(preprocess_tonic_spikes(delay(events), dataset.ordering,
                                              (dataset.sensor_size[0]*N_DELAY,
                                              dataset.sensor_size[1],
                                              dataset.sensor_size[2])))
    labels_val.append(label)

max_spikes = max(max_spikes, calc_max_spikes(spikes_val))
latest_spike_time = max(latest_spike_time, calc_latest_spike_time(spikes_val))

dataset = SHD(save_to="../data", train=False)

spikes_test = []
labels_test = []
for i in range(len(dataset)):
    events, label = dataset[i]
    events = np.delete(events, np.where(events["t"] >= 1000000))
    spikes_test.append(preprocess_tonic_spikes(delay(events), dataset.ordering,
                                            (dataset.sensor_size[0]*N_DELAY,
                                            dataset.sensor_size[1],
                                            dataset.sensor_size[2])))
    labels_test.append(label)

max_spikes = max(max_spikes, calc_max_spikes(spikes_test))
latest_spike_time = max(latest_spike_time, calc_latest_spike_time(spikes_test))

print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")



serialiser = Numpy("shd_augment_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes * N_DELAY),
                       num_input *  N_DELAY)
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output)

    # Connections
    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02)),
               Exponential(5.0))
    Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))

max_example_timesteps = int(np.ceil(latest_spike_time / DT))

compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses="sparse_categorical_crossentropy",
                                reg_lambda_upper=5e-11, reg_lambda_lower=5e-11,
                                reg_nu_upper=14, max_spikes=1500,
                                optimiser=Adam(0.001 * 0.01), batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)
best_acc, best_e = 0, 0
with compiled_net:
    # Loop through epochs
    start_time = perf_counter()
    callbacks = [Checkpoint(serialiser), SpikeRecorder(hidden, key="hidden_spikes", record_counts=True), EaseInSchedule()]

    for e in range(NUM_EPOCHS):
        # Apply augmentation to events and preprocess
        spikes_train = []
        labels_train = []
        blended_dataset = blend(raw_dataset, classes)
        for events, label in blended_dataset:
            spikes_train.append(preprocess_tonic_spikes(delay(shift(events)), dataset.ordering,
                                                    (dataset.sensor_size[0]*N_DELAY,
                                                    dataset.sensor_size[1],dataset.sensor_size[2])))
            labels_train.append(label)

        # Train epoch
        train_metrics, valid_metrics, train_cb, valid_cb  = compiled_net.train(
            {input: spikes_train}, {output: labels_train},
            start_epoch=e, num_epochs=1, shuffle=True,
            callbacks=callbacks, validation_callbacks=[],
            validation_x={input: spikes_val}, validation_y={output: labels_val})

        hidden_spikes = np.zeros(NUM_HIDDEN)
        for cb_d in train_cb['hidden_spikes']:
            hidden_spikes += cb_d
        print("Epoch", e, " Silent neurons: ", np.count_nonzero(hidden_spikes==0), " Training accuracy: ", 100 * train_metrics[output].result, " Validation accuracy: ", 100 * valid_metrics[output].result)
        hidden_sg = compiled_net.connection_populations[Conn_Pop0_Pop1]
        hidden_sg.vars["g"].pull_from_device()
        g_view = hidden_sg.vars["g"].view.reshape((num_input*N_DELAY, NUM_HIDDEN))
        g_view[:,hidden_spikes==0] += 0.002
        hidden_sg.vars["g"].push_to_device()
        if valid_metrics[output].result > best_acc:
            best_acc = valid_metrics[output].result
            best_e = e

    end_time = perf_counter()


    print(f"Training accuracy = {100 * train_metrics[output].result}%")
    print(f"Validation accuracy = {100 * valid_metrics[output].result}%")


# Load network state from final checkpoint
network.load((best_e,), serialiser)

compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                            reset_in_syn_between_batches=True,
                            batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)

with compiled_net:
    # Evaluate model on numpy dataset
    test_metrics, _  = compiled_net.evaluate({input: spikes_test},
                                        {output: labels_test})
    end_time = perf_counter()

    print(f"Test Accuracy = {100 * test_metrics[output].result}%")
    print(f"Time = {end_time - start_time}s")

    
