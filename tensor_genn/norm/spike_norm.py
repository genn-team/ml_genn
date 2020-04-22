import numpy as np 
from tqdm import trange

'''
References: 
A. Sengupta, Y. Ye, R. Wang, C, Liu, and K. Roy. 2019. Going Deeper in Spiking Neural Networks:
VGG and Residual Architectures. Frontiers in Neuroscience, 2019 (vol 13).
'''

class SpikeNorm(object):
    def __init__(self, norm_data, classify_time=500.0):
        self.norm_data = norm_data
        self.classify_time = classify_time

    def normalize(self, tg_model):
        print('Spike-Norm')
        g_model = tg_model.g_model
        layer_names = tg_model.layer_names

        # Set layer thresholds high initially
        for l in range(len(layer_names)):
            neurons = g_model.neuron_populations[layer_names[l] + '_nrn']
            neurons.extra_global_params['Vthr'].view[:] = np.inf

        # For each weighted layer
        for l in range(len(layer_names)):
            neurons = g_model.neuron_populations[layer_names[l] + '_nrn']
            neurons.extra_global_params['Vthr'].view[:] = np.float64(1.0)
            layer_threshold = np.float64(0.0)

            # For each sample presentation
            progress = trange(self.norm_data.shape[0])
            progress.set_description('layer <{}>'.format(layer_names[l]))
            for i in progress:

                # Set new input
                tg_model.reset_state()
                tg_model.set_inputs(self.norm_data[i])

                # Main simulation loop
                while g_model.t < self.classify_time:
                    tg_model.step_time()

                    g_model.pull_var_from_device(neurons.name, 'Vmem_peak')
                    Vmem_peak = neurons.vars['Vmem_peak'].view
                    layer_threshold = np.max([layer_threshold, Vmem_peak.max()])

            # Update this neuron population's threshold
            neurons.extra_global_params['Vthr'].view[:] = layer_threshold
            print('layer <{}> threshold: {}'.format(layer_names[l], layer_threshold))
