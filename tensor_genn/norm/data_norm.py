from math import ceil
import numpy as np 
import tensorflow.keras.backend as K

'''
References: 
Peter U. Diehl, Daniel Neil, Jonathan Binas, Matthew Cook, Shih-Chii Liu, and Michael Pfeiffer. 2015. 
Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing. IJCNN (2015)
'''

class DataNorm(object):
    def __init__(self, norm_samples, batch_size=None):
        self.norm_samples = norm_samples
        if batch_size == None:
            self.batch_size = len(norm_samples)
        else:
            self.batch_size = batch_size

    def normalize(self, tg_model):
        print('Data Norm')
        tf_model = tg_model.tf_model
        g_model = tg_model.g_model

        # Get output functions for weighted layers.
        idx = [i for i, l in enumerate(tf_model.layers) if l.get_weights() != []]
        get_outputs = K.function([tf_model.input], [tf_model.layers[i].output for i in idx])

        # Find the maximum activation in each layer, given input data.
        activation = np.empty(len(idx), dtype=np.float64)
        max_activation = np.zeros(len(idx), dtype=np.float64)
        n_batches = ceil(len(self.norm_samples) / self.batch_size)
        for i in range(n_batches):
            if i < n_batches - 1:
                x = self.norm_samples[i*self.batch_size : (i+1)*self.batch_size]
            else:
                x = self.norm_samples[i*self.batch_size : ]
            activation[:] = [np.max(out) for out in get_outputs(x)]
            max_activation[:] = np.max([max_activation, activation], 0)

        # Find the maximum weight in each layer.
        weights = tf_model.get_weights()
        max_weights = np.array([np.max(w) for w in weights], dtype=np.float64)

        # Compute scale factors and normalize weights.
        scale_factors = np.max([max_activation, max_weights], 0)
        applied_factors = np.empty(len(idx))
        applied_factors[0] = scale_factors[0]
        applied_factors[1:] = scale_factors[1:] / scale_factors[:-1]

        # Update this neuron population's threshold
        for i, layer_name in enumerate([tf_model.layers[i].name for i in idx]):
            neurons = g_model.neuron_populations[layer_name + '_nrn']
            neurons.extra_global_params['Vthr'].view[:] = applied_factors[i]
