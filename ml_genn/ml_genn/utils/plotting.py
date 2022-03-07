import matplotlib.pyplot as plt 
import math

def raster_plot(spike_ids, spike_times, neuron_pops, time=None):
    for st, si in zip(spike_times, spike_ids):
        fig, ax = plt.subplots(math.ceil(len(neuron_pops) / 3.0), 3, sharex="col")
        ax = trim_ax(ax, len(neuron_pops))
        for ax, (j, npop) in zip(ax, enumerate(neuron_pops)):
            ax.set_title(npop.name + ' ' + str(len(st[j])))
            ax.scatter(st[j], si[j], s=0.3)
            ax.set_ylim((0, npop.size))
            if time is not None:
                ax.set_xlim((0, time))
    plt.show()

def trim_ax(ax, N):
    ax = ax.flat
    for a in ax[N:]:
        a.remove()
    return ax[:N]
