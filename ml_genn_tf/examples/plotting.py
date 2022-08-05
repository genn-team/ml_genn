import matplotlib.pyplot as plt

from ml_genn.callbacks import SpikeRecorder
from typing import Sequence

def plot_spikes(callbacks: Sequence, plot_sample_spikes: Sequence, show: bool = True):
    # Filter out spike recorder callbacks
    spike_recorder_callbacks = [c for c in callbacks 
                                if isinstance(c, SpikeRecorder)]
                    
    # Create figure with rows of axes for spike recorders 
    # and columns for samples we want to plot
    fig, axes = plt.subplots(len(spike_recorder_callbacks), 
                             len(plot_sample_spikes), 
                             sharex="col", sharey="row")
    
    # Loop through spike recorder callbacks
    for i, c in enumerate(spike_recorder_callbacks):
        # Extract spike times and IDs from callback
        spike_times, spike_ids = c.spikes
        
        # If we're only plotting one sample, plot
        if len(plot_sample_spikes) == 1:
            s = plot_sample_spikes[0]
            axes[i].scatter(spike_times[s], spike_ids[s], s=2)
            axes[i].set_ylabel("Neuron")
        # Otherwise, loop through samples and plot
        else:
            axes[i, 0].set_ylabel("Neuron")
            for j, s in enumerate(plot_sample_spikes):
                axes[i, j].scatter(spike_times[j], spike_ids[j], s=2)
    
    # Show if desired
    if show:
        plt.show()
    
    return fig, axes
