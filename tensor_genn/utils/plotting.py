import matplotlib.pyplot as plt 
import math

def raster_plot(spike_ids, spike_times, neuron_pops):
    for st,si in zip(spike_times,spike_ids):
        fig, axs = plt.subplots(math.ceil(len(neuron_pops)/3.),3)
        axs = trim_axs(axs,len(neuron_pops))
        for ax,(j,npop) in zip(axs,enumerate(neuron_pops)):
            ax.set_title(npop.name + ' ' + str(len(st[j])))
            ax.scatter(st[j],si[j],s=0.3)
    plt.show()

def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]