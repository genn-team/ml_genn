import tensorflow.keras.backend as K
import numpy as np

'''
References: 
Peter U. Diehl, Daniel Neil, Jonathan Binas, Matthew Cook, Shih-Chii Liu, and Michael Pfeiffer. 2015. 
Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing. IJCNN (2015)
'''

class DataNorm(object):
    def __init__(self, norm_data, tf_model):
        self.norm_data = norm_data
        self.tf_model = tf_model

    def normalize(self, mlg_model):
        print('Data-Norm')
        g_model = mlg_model.g_model

        # Get output functions for weighted layers.
        idx = [i for i, l in enumerate(self.tf_model.layers) if l.get_weights() != []]
        get_outputs = K.function(self.tf_model.inputs, [self.tf_model.layers[i].output for i in idx])

        # Find the maximum activation in each layer, given input data.
        max_activation = np.array([np.max(out) for out in get_outputs(self.norm_data)], dtype=np.float64)

        # Find the maximum weight in each layer.
        max_weights = np.array([np.max(w) for w in self.tf_model.get_weights()], dtype=np.float64)

        # Compute scale factors and normalize weights.
        scale_factors = np.max([max_activation, max_weights], 0)
        applied_factors = np.empty(scale_factors.shape, dtype=np.float64)
        applied_factors[0] = scale_factors[0]
        applied_factors[1:] = scale_factors[1:] / scale_factors[:-1]

        # Update layer thresholds
        for layer, threshold in zip(mlg_model.layers[1:], applied_factors):
            print('layer <{}> threshold: {}'.format(layer.name, threshold))
            layer.neurons.set_threshold(threshold)
