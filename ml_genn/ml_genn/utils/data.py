import numpy as np

from collections import namedtuple
from typing import Optional, Sequence, Union
from ..metrics import Metric

from copy import deepcopy

from .module import get_object

MetricType = Union[Metric, str]
MetricsType = Union[dict, MetricType]


PreprocessedSpikes = namedtuple("PreprocessedSpikes", ["end_spikes", "spike_times"])


def get_dataset_size(data) -> Optional[int]:
    sizes = [len(d) for d in data.values()]
    return sizes[0] if len(set(sizes)) <= 1 else None


def batch_dataset(data, batch_size, size):
    # Perform split, resulting in {key: split data} dictionary
    splits = range(0, size, batch_size)
    data = {k: [v[s:s + batch_size] for s in splits]
            for k, v in data.items()}

    # Create list with dictionary for each split
    data_list = [{} for _ in splits]

    # Loop through batches of data
    for k, batches in data.items():
        # Copy batches of data into dictionaries
        for d, b in zip(data_list, batches):
            d[k] = b

    return data_list

def permute_dataset(data, indices):
    # Check indices are correct shape
    assert len(indices) == get_dataset_size(data)

    # Shuffle each value, using numpy fa
    shuffled_data = {}
    for k, v in data.items():
        # **TODO** better type check
        if isinstance(v, np.ndarray):
            shuffled_data[k] = v[indices]
        else:
            shuffled_data[k] = [v[i] for i in indices]
    
    return shuffled_data

def preprocess_spikes(times, ids, num_neurons):
    # Calculate end spikes
    end_spikes = np.cumsum(np.bincount(ids, minlength=num_neurons))

    # Sort events first by neuron id and then 
    # by time and use to order spike times
    times = times[np.lexsort((times, ids))]

    # Return end spike indices and spike times
    return PreprocessedSpikes(end_spikes, times)

 
# **TODO** maybe this could be a static from_tonic method 
def preprocess_tonic_spikes(events, ordering, shape, time_scale=1.0 / 1000.0):
    # Calculate cumulative sum of each neuron's spike count
    num_neurons = np.product(shape) 

    # Check dataset datatype includes time and polarity
    if "t" not in ordering or "p" not in ordering:
        raise RuntimeError("Only tonic datasets with time (t) and "
                           "polarity (p) in ordering are supported")

    # If sensor has single polarity
    if shape[2] == 1:
        # If sensor is 2D, flatten x and y into event IDs
        if ("x" in ordering) and ("y" in ordering):
            spike_event_ids = events["x"] + (events["y"] * shape[1])
        # Otherwise, if it's 1D, simply use X
        elif "x" in ordering:
            spike_event_ids = events["x"]
        else:
            raise RuntimeError("Only 1D and 2D sensors supported")
    # Otherwise
    else:
        # If sensor is 2D, flatten x, y and p into event IDs
        if ("x" in ordering) and ("y" in ordering):
            spike_event_ids = (events["p"] +
                               (events["x"] * shape[2]) + 
                               (events["y"] * shape[1] * shape[2]))
        # Otherwise, if it's 1D, flatten x and p into event IDs
        elif "x" in ordering:
            spike_event_ids = events["p"] + (events["x"] * shape[2])
        else:
            raise RuntimeError("Only 1D and 2D sensors supported")
    
    # Preprocess scaled times and flattened IDs
    return preprocess_spikes(events["t"] * time_scale, spike_event_ids,
                             num_neurons)

def log_latency_encode_data(data, tau_eff, thresh, max_stimuli_time=None):
    # Loop through examples
    spikes = []
    for i in range(len(data)):
        # Get boolean mask of spiking neurons
        spike_vector = data[i] > thresh
        
        # Take cumulative sum to get end spikes
        end_spikes = np.cumsum(spike_vector)
            
        # Extract values of spiking pixels
        spike_pixels = data[i,spike_vector]
        
        # Calculate spike times
        spike_times = tau_eff * np.log(spike_pixels / (spike_pixels - thresh))
        
        # Add to list
        spikes.append(PreprocessedSpikes(end_spikes, spike_times))

    return spikes

def batch_spikes(spikes: Sequence[PreprocessedSpikes], batch_size: int):
    # Check that there aren't more examples than batch size 
    # and that all examples are for same number of neurons
    num_neurons = len(spikes[0].end_spikes)
    if len(spikes) > batch_size:
        raise RuntimeError(f"Cannot batch {len(spikes)} PreprocessedSpikes "
                           f"when batch size is only {batch_size}")
    if any(len(s.end_spikes) != num_neurons for s in spikes):
        raise RuntimeError("Cannot batch PreprocessedSpikes "
                           "with different numbers of neurons")
        
    assert all(len(s.end_spikes) == num_neurons for s in spikes)

    # Extract seperate lists of each example's
    # end spike indices and spike times
    end_spikes, spike_times = zip(*spikes)

    # Calculate cumulative sum of spikes counts across batch
    cum_spikes_per_example = np.concatenate(
        ([0], np.cumsum([len(s) for s in spike_times])))

    # Add this cumulative sum onto the end spikes array of each example
    # **NOTE** zip will stop before extra cum_spikes_per_example value
    batch_end_spikes = np.vstack(
        [c + e for e, c in zip(end_spikes, cum_spikes_per_example)])

    # If this isn't a full batch
    if len(spikes) < batch_size:
        # Create spike padding for remainder of batch
        pad_shape = (batch_size - len(spikes), num_neurons)
        spike_padding = np.ones(pad_shape, dtype=int) * cum_spikes_per_example[-1]

        # Stack onto end spikes
        batch_end_spikes = np.vstack((batch_end_spikes, spike_padding))

    # Concatenate together all spike times
    batch_spike_times = np.concatenate(spike_times)

    return PreprocessedSpikes(batch_end_spikes, batch_spike_times)

def calc_start_spikes(end_spikes):
    start_spikes = np.empty_like(end_spikes)
    if end_spikes.ndim == 1:
        start_spikes[0] = 0
        start_spikes[1:] = end_spikes[:-1]
    else:
        start_spikes[0, 0] = 0
        start_spikes[1:, 0] = end_spikes[:-1, -1]
        start_spikes[:, 1:] = end_spikes[:, :-1]

    return start_spikes

def calc_max_spikes(spikes):
    return max(len(p.spike_times) for p in spikes)

def calc_latest_spike_time(spikes):
    return max(max(p.spike_times) for p in spikes)