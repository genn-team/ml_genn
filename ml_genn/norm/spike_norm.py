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

    def normalize(self, mlg_model, time):
        print('Spike-Norm')
        g_model = mlg_model.g_model
        n_samples = self.norm_data[0].shape[0]

        # Set layer thresholds high initially
        for layer in mlg_model.layers[1:]:
            layer.neurons.set_threshold(np.inf)

        # For each weighted layer
        for layer in mlg_model.layers[1:]:
            threshold = np.float64(0.0)

            # For each sample presentation
            progress = tqdm(total=n_samples)
            for batch_start in range(0, n_samples, mlg_model.batch_size):
                batch_end = min(batch_start + mlg_model.batch_size, n_samples)
                batch_data = [x[batch_start:batch_end] for x in self.norm_data]

                # Set new input
                mlg_model.reset()
                mlg_model.set_input_batch(batch_data)

                # Main simulation loop
                while g_model.t < time:

                    # Step time
                    mlg_model.step_time()

                    # Get maximum activation
                    for batch_i in range(batch_end - batch_start):
                        nrn = layer.neurons.nrn[batch_i]
                        nrn.pull_var_from_device('Vmem')
                        threshold = np.max([threshold, nrn.vars['Vmem'].view.max()])
                        nrn.vars['Vmem'].view[:] = np.float64(0.0)
                        nrn.push_var_to_device('Vmem')

                progress.update(batch_end - batch_start)

            progress.close()

            # Update this layer's threshold
            print('layer <{}> threshold: {}'.format(layer.name, threshold))
            layer.neurons.set_threshold(threshold)
