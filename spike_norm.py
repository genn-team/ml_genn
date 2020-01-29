import numpy as np 
import math

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D

'''
References: 
Peter U. Diehl, Daniel Neil, Jonathan Binas, Matthew Cook, Shih-Chii Liu, and Michael Pfeiffer. 2015. 
Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing. IJCNN (2015)
'''

class SpikeNorm():
    def __init__(self, data, present_time=100.0):
        self.data = data
        self.present_time = present_time

    '''
    The maximum activation encountered in the training data is a good estimate of the highest
    activation possible in the model. This 'scale factor' is computed separately for each layer 
    of the network.
    The 'applied factor' is the scaled factor of each layer, divided by that of the previous layer. This
    brings all the factors to a uniform range of values (with respect to the input layer).
    Weights of each layer in the model are then divided by the corresponding applied factor to compute the
    final normalized weights.
    '''

    '''
    v_th_norm[0 : n_layers] = 0
    layer[0].input = spikes
    for i in range( n_layers ):
        for t in range( n_timesteps ):
            layer[i].forward( layer[i].input[t] )
            v_th_norm[i] = max( v_th_norm[i],  max( dot( layer[i].weights, layer[i].input[t] ) ) )

        layer[i].v_th = v_th_norm[i]
        layer[i+1].input = layer[i].forward( layer[i].input )
    '''

    def normalize(self, tg_model):
        print('Spike Norm')
        genn_model = tg_model.genn_model
        scale_factors = np.zeros(len(tg_model.layer_names))

        # For each synapse population
        for i, name in enumerate(tg_model.layer_names):
            neurons = genn_model.neuron_populations[name + '_nrn']

            # CONV2D LAYERS ONLY
            #if isinstance(genn_model.neuron_populations[neurons.name], Conv2D):
            #    continue

            # For each sample
            for x in self.data:

                # Before simulation
                for nrn in genn_model.neuron_populations.values():
                    nrn.vars['Vmem'].view[:] = 0.0
                    nrn.vars['Vmem_peak'].view[:] = 0.0
                    nrn.vars['nSpk'].view[:] = 0
                    genn_model.push_state_to_device(nrn.name)


                # TODO: INPUT RATE ENCODING
                # FOR NOW, USE CONSTANT CURRENT INJECTION EQUAL TO INPUT MAGNITUDE
                genn_model.current_sources['input_cs'].vars['magnitude'].view[:] = x.flatten()
                genn_model.push_var_to_device('input_cs', 'magnitude')


                # Run simulation
                for t in range(math.ceil(self.present_time / genn_model.dT)):
                    genn_model.step_time()

                    genn_model.pull_var_from_device(neurons.name, 'Vmem_peak')
                    Vmem_peak = neurons.vars['Vmem_peak'].view
                    scale_factors[i] = np.max([scale_factors[i], Vmem_peak.max()])

            # Update this neuron population's threshold
            neurons.extra_global_params['Vthr'].view[:] = scale_factors[i]

            # # Update this synapse population's weights
            # synapses = genn_model.synapse_populations[name + '_syn']
            # synapses.vars['g'].view[:] /= scale_factors[i]
            # genn_model.push_var_to_device(synapses.name, 'g')

            print('layer: ' + name)
            print(scale_factors)
