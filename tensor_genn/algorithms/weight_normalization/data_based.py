import numpy as np 
import math

import tensorflow as tf 
import tensorflow.keras.backend as K
from pygenn import genn_model, genn_wrapper

'''
References: 
Peter U. Diehl, Daniel Neil, Jonathan Binas, Matthew Cook, Shih-Chii Liu, and Michael Pfeiffer. 2015. 
Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing. IJCNN (2015)
'''

class DataBased():
    def __init__(self,data=None, batch_size=100):
        self.x_train = data
        self.batch_size = batch_size # Adjust based on available system memory

    def normalize(self,tf_model):
        """
        The maximum activation encountered in the training data is a good estimate of the highest
        activation possible in the model. This 'scale factor' is computed separately for each layer 
        of the network.
        The 'applied factor' is the scaled factor of each layer, divided by that of the previous layer. This
        brings all the factors to a uniform range of values (with respect to the input layer).
        Weights of each layer in the model are then divided by the corresponding applied factor to compute the
        final normalized weights.
        Only Convolution2D layer weights are normalized.
        """
        # Get parameterized layer indices
        wts_exist_inds = [i for i,layer in enumerate(tf_model.layers) if len(layer.get_weights()) > 0 and isinstance(layer,tf.keras.layers.Conv2D)]
        
        # Function to fetch ReLU outputs from intermediate layers
        get_output_from_layer = K.function([tf_model.layers[0].input],
                                           [tf_model.layers[i].output for i in wts_exist_inds])

        n_batches = math.ceil(len(self.x_train)/self.batch_size)
        max_wts = [np.max(wt) for wt in [tf_model.layers[i].get_weights() for i in wts_exist_inds]]
        max_acts = [0]*len(wts_exist_inds)
        for bi in range(n_batches):
            # get training set batch
            if bi < n_batches - 1:
                x = self.x_train[bi*self.batch_size:(bi+1)*self.batch_size]
            else:
                x = self.x_train[bi*self.batch_size:]

            # maximum output activations for each layer
            layer_outputs = get_output_from_layer([x])
            b_macts = [np.max(op) for op in layer_outputs]
            max_acts = [max(b, a) for b, a in zip(b_macts, max_acts)]
        
        # Compute scale factors and normalize weights
        scale_factors = [max(a, w) for a, w in zip(max_acts, max_wts)] # 2
        applied_factors = [scale_factors[0]] + [scale_factors[i]/scale_factors[i-1] for i in range(1,len(scale_factors))]
        scaled_weights = tf_model.get_weights()
        for wei,i in zip(wts_exist_inds,range(len(wts_exist_inds))):
            scaled_weights[wei] /= applied_factors[i]
        
        return scaled_weights