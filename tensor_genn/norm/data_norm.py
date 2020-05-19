import tensorflow.keras.backend as K
import numpy as np

'''
References: 
Peter U. Diehl, Daniel Neil, Jonathan Binas, Matthew Cook, Shih-Chii Liu, and Michael Pfeiffer. 2015. 
Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing. IJCNN (2015)
'''

class DataNorm(object):
    def __init__(self, norm_samples, tf_model):
        self.norm_samples = norm_samples
        self.tf_model = tf_model

    def normalize(self, tg_model):
        print('Data-Norm')
        g_model = tg_model.g_model

        # Get output functions for weighted layers.
        idx = [i for i, l in enumerate(self.tf_model.layers) if l.get_weights() != []]
        get_outputs = K.function([self.tf_model.input], [self.tf_model.layers[i].output for i in idx])

        # Find the maximum activation in each layer, given input data.
        max_activation = np.array([np.max(out) for out in get_outputs(self.norm_samples)], dtype=np.float64)

        # Find the maximum weight in each layer.
        max_weights = np.array([np.max(w) for w in self.tf_model.get_weights()], dtype=np.float64)

        # Compute scale factors and normalize weights.
        scale_factors = np.max([max_activation, max_weights], 0)
        applied_factors = np.empty(len(idx))
        applied_factors[0] = scale_factors[0]
        applied_factors[1:] = scale_factors[1:] / scale_factors[:-1]

        # Update layer thresholds
        for l in range(len(tg_model.layer_names)):
            print('layer <{}> threshold: {}'.format(tg_model.layer_names[l], applied_factors[l]))
            tg_model.thresholds[l] = applied_factors[l]
            for batch_i in range(tg_model.batch_size):
                name = tg_model.layer_names[l] + '_nrn_' + str(batch_i)
                nrn = g_model.neuron_populations[name]
                nrn.extra_global_params['Vthr'].view[:] = applied_factors[l]
