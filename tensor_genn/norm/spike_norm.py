import numpy as np 
from tqdm import tqdm

'''
References: 
A. Sengupta, Y. Ye, R. Wang, C, Liu, and K. Roy. 2019. Going Deeper in Spiking Neural Networks:
VGG and Residual Architectures. Frontiers in Neuroscience, 2019 (vol 13).
'''

class SpikeNorm(object):
    def __init__(self, norm_data):
        self.norm_data = norm_data

    def normalize(self, tg_model, time):
        print('Spike-Norm')
        g_model = tg_model.g_model
        n_samples = self.norm_data.shape[0]

        # Set all layer thresholds high initially
        for l in range(len(tg_model.layer_names)):
            for batch_i in range(tg_model.batch_size):
                name = tg_model.layer_names[l] + '_nrn_' + str(batch_i)
                nrn = g_model.neuron_populations[name]
                nrn.extra_global_params['Vthr'].view[:] = np.inf

        # For each weighted layer
        for l in range(len(tg_model.layer_names)):
            threshold = np.float64(0.0)

            # For each sample presentation
            progress = tqdm(total=n_samples)
            for batch_start in range(0, n_samples, tg_model.batch_size):
                batch_end = min(batch_start + tg_model.batch_size, n_samples)
                batch_norm_data = self.norm_data[batch_start:batch_end]

                # Set new input
                tg_model.reset_state()
                tg_model.set_input_batch(batch_norm_data)

                # Main simulation loop
                while g_model.t < time:

                    # Step time
                    tg_model.step_time()

                    # Get maximum activation
                    for batch_i in range(batch_end - batch_start):
                        name = tg_model.layer_names[l] + '_nrn_' + str(batch_i)
                        nrn = g_model.neuron_populations[name]
                        nrn.pull_var_from_device('Vmem')
                        threshold = np.max([threshold, nrn.vars['Vmem'].view.max()])
                        nrn.vars['Vmem'].view[:] = np.float64(0.0)
                        nrn.push_var_to_device('Vmem')

                progress.update(batch_end - batch_start)

            progress.close()

            # Update this layer's threshold
            print('layer <{}> threshold: {}'.format(tg_model.layer_names[l], threshold))
            tg_model.thresholds[l] = threshold
            for batch_i in range(tg_model.batch_size):
                name = tg_model.layer_names[l] + '_nrn_' + str(batch_i)
                nrn = g_model.neuron_populations[name]
                nrn.extra_global_params['Vthr'].view[:] = threshold
