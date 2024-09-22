import numpy as np

from collections import namedtuple
from typing import (Any, Dict, List, Mapping, Optional, 
                    Sequence, Sized, Tuple, Union)
from ..metrics import Metric

from copy import deepcopy

from .module import get_object

DataDictType = Mapping[Any, Union[Sequence, np.ndarray]]

PreprocessedSpikes = namedtuple("PreprocessedSpikes", ["end_spikes", "spike_times"])

def split_dataset(data: DataDictType, split: float) -> Tuple[DataDictType,
                                                             DataDictType]:
    # Check that split is valid
    if split < 0.0 or split > 1.0:
        raise RuntimeError(f"Invalid split of {split}")

    # Get size of dataset
    dataset_size = get_dataset_size(data)

    # Get point where dataset is split
    split_point = int(round((1.0 - split) * dataset_size))

    # Return two dictionaries with values from before and after split point
    return ({k: v[:split_point] for k, v in data.items()},
            {k: v[split_point:] for k, v in data.items()})

def get_dataset_size(data: DataDictType) -> Optional[int]:
    sizes = [len(d) for d in data.values()]
    return sizes[0] if len(set(sizes)) <= 1 else None

def batch_dataset(data: DataDictType, batch_size: int, size: int):
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

def permute_dataset(data: DataDictType, 
                    indices: Union[Sequence[int], np.ndarray]):
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

def preprocess_spikes(times: np.ndarray, ids: np.ndarray,
                      num_neurons: int) -> PreprocessedSpikes:
    # Calculate end spikes
    end_spikes = np.cumsum(np.bincount(ids, minlength=num_neurons))

    # Sort events first by neuron id and then 
    # by time and use to order spike times
    times = times[np.lexsort((times, ids))]

    # Return end spike indices and spike times
    return PreprocessedSpikes(end_spikes, times)

# **TODO** maybe this could be a static from_tonic method 
def preprocess_tonic_spikes(events: np.ndarray, ordering: Sequence[str],
                            shape: Tuple, time_scale=1.0 / 1000.0,
                            dt: Optional[float] = None,
                            histogram_thresh : Optional[int] = None) -> PreprocessedSpikes:
    """Preprocess a Tonic format spike train into PreprocessedSpikes format

    Args:
        events:             Structured array containing events
        ordering:           Names of fields in events array
        shape:              Shape of sensor events came from
        time_scale:         Scale to apply to event times, typically to 
                            convert from microseconds to milliseconds
        dt:                 Timestep to discretise events to
        histogram_thresh:   Minimum number of source events required to
                            trigger spike in downsampled dt
    """
    # Calculate cumulative sum of each neuron's spike count
    num_neurons = np.prod(shape) 

    # Check dataset datatype includes time and polarity
    if "t" not in ordering or "p" not in ordering:
        raise RuntimeError("Only tonic datasets with time (t) and "
                           "polarity (p) in ordering are supported")

    # If sensor has single polarity
    if shape[2] == 1:
        # If sensor is 2D, flatten x and y into event IDs
        if ("x" in ordering) and ("y" in ordering):
            spike_event_ids = events["x"] + (events["y"] * shape[0])
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
                               (events["y"] * shape[0] * shape[2]))
        # Otherwise, if it's 1D, flatten x and p into event IDs
        elif "x" in ordering:
            spike_event_ids = events["p"] + (events["x"] * shape[2])
        else:
            raise RuntimeError("Only 1D and 2D sensors supported")
    
    scaled_t = events["t"] * time_scale
    if histogram_thresh is None:
        return preprocess_spikes(scaled_t, spike_event_ids,
                                 num_neurons)
    else:
        # Build ranges for neuron ids and timesteps
        assert dt is not None
        neuron_range = np.arange(num_neurons + 1)
        timestep_range = np.arange(0.0, np.amax(scaled_t) + dt, dt)

        # Compute histogram
        spike_event_hist = np.histogram2d(spike_event_ids, scaled_t,
                                          (neuron_range, timestep_range))[0]

        # Find indices of bins where there are enough events
        thresh_id, thresh_t = np.where(spike_event_hist >= histogram_thresh)

        # Preprocess
        return preprocess_spikes(thresh_t * dt, thresh_id, num_neurons)

def generate_yin_yang_dataset(size: int, scale: float, offset: float=0.0,
                              bias: bool = True, r_small: float = 0.1,
                              r_big: float = 0.5):
    """Generate Yin-Yang dataset [Kriener2021]_ in PreprocessedSpikes format.
    Code copied from https://github.com/lkriener/yin_yang_data_set

    Args:
        size:       Number of Yin-Yang stimuli to generate
        scale:      Scale factor to apply to spike times
        offset:     Minimum spike time in ms
        bias:       Should an additional fixed bias spike at offset 
                    time be added to each example
        r_small:    Radius of small (eye) circles
        r_big:      Radius of large circles
    """
    def dist_to_right_dot(x, y):
        return np.sqrt((x - 1.5 * r_big)**2 + (y - r_big)**2)

    def dist_to_left_dot(x, y):
        return np.sqrt((x - 0.5 * r_big)**2 + (y - r_big)**2)

    def which_class(x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = dist_to_right_dot(x, y)
        d_left = dist_to_left_dot(x, y)
        criterion1 = d_right <= r_small
        criterion2 = d_left > r_small and d_left <= 0.5 * r_big
        criterion3 = y > r_big and d_right > 0.5 * r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < r_small or d_left < r_small
        if is_circles:
            return 2
        return int(is_yin)

    def get_sample(goal):
        # sample until goal is satisfied
        while True:
            # sample x,y coordinates
            x, y = np.random.random_sample(2) * 2. * r_big
            # check if within yin-yang circle
            if np.sqrt((x - r_big)**2 + (y - r_big)**2) > r_big:
                continue
            # check if they have the same class as the goal for this sample
            if which_class(x, y) == goal:
                return x, y

    # Loop through number of stimuli to generate
    spikes = []
    labels = []
    num_channels = 5 if bias else 4
    ids = np.arange(num_channels)
    for i in range(size):
        # keep num of class instances balanced by using rejection sampling
        # choose class for this sample
        goal_class = np.random.randint(3)
        x, y = get_sample(goal_class)
        
        
        # Generate vector of 4 values and add bias if desired
        times = [x, y, 1.0 - x, 1.0 - y]
        if bias:
            times.insert(0, 0.0)
    
        spikes.append(
            preprocess_spikes(offset + (np.asarray(times) * scale),
                              ids, num_channels))
        labels.append(goal_class)

    return spikes, labels

def linear_latency_encode_data(data: np.ndarray, max_time: float,
                               min_time: float = 0.0,
                               thresh: int = 1) -> List[PreprocessedSpikes]:
    """Generate PreprocessedSpikes format stimuli by linearly
    latency encoding static data

    Args:
        data:       Data in uint8 format
        max_time:   Spike time for inputs with a value of 0
        min_time:   Spike time for inputs with a value of 255
        thresh:     Threshold values must reach for any spike to be emitted
    """
    # **TODO** handle floating point data
    # Loop through examples
    time_range = max_time - min_time
    spikes = []
    for i in range(len(data)):
        # Get boolean mask of spiking neurons
        spike_vector = data[i] > thresh

        # Take cumulative sum to get end spikes
        end_spikes = np.cumsum(spike_vector)

        # Extract values of spiking pixels
        spike_pixels = data[i, spike_vector]

        # Calculate spike times
        spike_times = (((255.0 - spike_pixels) / 255.0) * time_range) + min_time

        # Add to list
        spikes.append(PreprocessedSpikes(end_spikes, spike_times))

    return spikes

def log_latency_encode_data(data: np.ndarray, tau_eff: float,
                            thresh: float) -> List[PreprocessedSpikes]:
    """Generate PreprocessedSpikes format stimuli by log-latency
    encoding static data

    .. math::

        T(x)= \\begin{cases}
                  \\tau_\\text{eff} log\\left(\\frac{x}{x - \\nu}\\right) & x > \\nu \\\\
                  \\infty & otherwise
              \\end{cases}

    Args:
        data:       Data in uint8 format
        tau_eff:    Scaling factor
        thresh:     Threshold values must reach for any spike to be emitted
    """
    # Loop through examples
    spikes = []
    for i in range(len(data)):
        # Get boolean mask of spiking neurons
        spike_vector = data[i] > thresh

        # Take cumulative sum to get end spikes
        end_spikes = np.cumsum(spike_vector)

        # Extract values of spiking pixels
        spike_pixels = data[i, spike_vector]

        # Calculate spike times
        spike_times = tau_eff * np.log(spike_pixels / (spike_pixels - thresh))

        # Add to list
        spikes.append(PreprocessedSpikes(end_spikes, spike_times))

    return spikes

def batch_spikes(spikes: Sequence[PreprocessedSpikes],
                 batch_size: int) -> PreprocessedSpikes:
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

def calc_start_spikes(end_spikes: np.ndarray) -> np.ndarray:
    start_spikes = np.empty_like(end_spikes)
    if end_spikes.ndim == 1:
        start_spikes[0] = 0
        start_spikes[1:] = end_spikes[:-1]
    else:
        start_spikes[0, 0] = 0
        start_spikes[1:, 0] = end_spikes[:-1, -1]
        start_spikes[:, 1:] = end_spikes[:, :-1]

    return start_spikes

def calc_max_spikes(spikes: PreprocessedSpikes) -> int:
    """Calculate maximum number of spikes emitted by any neuron

    Args:
        spikes: Spike train
    """
    return max(len(p.spike_times) for p in spikes)

def calc_latest_spike_time(spikes: PreprocessedSpikes) -> float:
    """Calculate time of the last spike emitted by any neuron

    Args:
        spikes: Spike train
    """
    return max(max(p.spike_times) for p in spikes)
