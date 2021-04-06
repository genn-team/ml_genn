import matplotlib.pyplot as plt 
import math

def raster_plot(spike_ids, spike_times, neuron_pops):
    for st, si in zip(spike_times, spike_ids):
        fig, ax = plt.subplots(math.ceil(len(neuron_pops) / 3.0), 3, sharex="col")
        ax = trim_ax(ax, len(neuron_pops))
        for ax, (j, npop) in zip(ax, enumerate(neuron_pops)):
            ax.set_title(npop.name + ' ' + str(len(st[j])))
            ax.scatter(st[j], si[j], s=0.3)
    plt.show()

def trim_ax(ax, N):
    ax = ax.flat
    for ax in ax[N:]:
        ax.remove()
    return ax[:N]
