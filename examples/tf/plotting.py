import matplotlib.pyplot as plt

from ml_genn.callbacks import SpikeRecorder
from typing import Sequence

def plot_spikes(cb_data: dict, plot_sample_spikes: Sequence, show: bool = True):
    # Create figure with rows of axes for spike recorders 
    # and columns for samples we want to plot
    fig, axes = plt.subplots(len(cb_data), 
                             len(plot_sample_spikes), 
                             sharex="col", sharey="row")
    
    # Loop through spike recorder callbacks
    for i, (key, data) in enumerate(cb_data.items()):
        # Extract spike times and IDs from callback
        spike_times, spike_ids = data
        
        assert len(plot_sample_spikes) == len(spike_times)
        
        # If we're only plotting one sample, plot
        if len(spike_times) == 1:
            axes[i].scatter(spike_times[0], spike_ids[0], s=2)
            axes[i].set_ylabel(key)
        # Otherwise, loop through samples and plot
        else:
            axes[i, 0].set_ylabel(key)
            for j, (time, id) in enumerate(zip(spike_times, spike_ids)):
                axes[i, j].scatter(time, id, s=2)
    
    # Label examples
    if len(plot_sample_spikes) == 1:
        axes[0].set_title(f"Example {plot_sample_spikes[0]}")
    else:
        for j, s in enumerate(plot_sample_spikes):
            axes[0, j].set_title(f"Example {s}")
    
    # Show if desired
    if show:
        plt.show()
    
    return fig, axes
