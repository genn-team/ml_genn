from math import ceil
import numpy as np 

'''
References: 
A. Sengupta, Y. Ye, R. Wang, C, Liu, and K. Roy. 2019. Going Deeper in Spiking Neural Networks:
VGG and Residual Architectures. Frontiers in Neuroscience, 2019 (vol 13).
'''

class SpikeNorm():
    def __init__(self, data, classify_time=500.0, classify_spikes=None):
        self.data = data
        self.classify_time = classify_time
        self.classify_spikes = classify_spikes

    def normalize(self, tg_model):
        print('Spike Norm')
        g_model = tg_model.g_model
        scale_factors = np.zeros(len(tg_model.layer_names))

        # For each synapse population
        for i, layer_name in enumerate(tg_model.layer_names):
            neurons = g_model.neuron_populations[layer_name + '_nrn']

            # For each sample
            for x in self.data:

                # Reset simulation state
                g_model._slm.initialize()

                # Set imulation inputs
                tg_model.set_genn_inputs(x)

                # Run simulation
                for t in range(ceil(self.classify_time / g_model.dT)):
                    g_model.step_time()

                    g_model.pull_var_from_device(neurons.name, 'Vmem_peak')
                    Vmem_peak = neurons.vars['Vmem_peak'].view
                    scale_factors[i] = np.max([scale_factors[i], Vmem_peak.max()])

                    # Break simulation if we have enough output spikes.
                    if self.classify_spikes is not None:
                        output_neurons = g_model.neuron_populations[tg_model.layer_names[-1] + '_nrn']
                        g_model.pull_var_from_device(output_neurons.name, 'nSpk')
                        if output_neurons.vars['nSpk'].view.sum() >= self.classify_spikes:
                            break

            # Update this neuron population's threshold
            neurons.extra_global_params['Vthr'].view[:] = scale_factors[i]

            print(layer_name)
            print(scale_factors)
