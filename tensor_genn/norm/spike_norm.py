import numpy as np 
from tqdm import trange

'''
References: 
A. Sengupta, Y. Ye, R. Wang, C, Liu, and K. Roy. 2019. Going Deeper in Spiking Neural Networks:
VGG and Residual Architectures. Frontiers in Neuroscience, 2019 (vol 13).
'''

class SpikeNorm(object):
    def __init__(self, norm_samples, classify_time=500.0):
        self.norm_samples = norm_samples
        self.classify_time = classify_time
        self.classify_spikes = classify_spikes

    def normalize(self, tg_model):
        print('Spike Norm')
        g_model = tg_model.g_model
        scale_factors = np.zeros(len(tg_model.layer_names))

        # For each weighted layer
        for l, layer_name in enumerate(tg_model.layer_names):
            neurons = g_model.neuron_populations[layer_name + '_nrn']

            # For each sample
            for x in self.norm_samples:

                # Reset state
                tg_model.reset_state()

                # Set inputs
                tg_model.set_inputs(x)

                # Main simulation loop
                while g_model.t < self.classify_time:
                    tg_model.step_time()

                    g_model.pull_var_from_device(neurons.name, 'Vmem_peak')
                    Vmem_peak = neurons.vars['Vmem_peak'].view
                    scale_factors[l] = np.max([scale_factors[l], Vmem_peak.max()])

            # Update this neuron population's threshold
            neurons.extra_global_params['Vthr'].view[:] = scale_factors[l]

            print(layer_name)
            print(scale_factors)
