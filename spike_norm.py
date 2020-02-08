import numpy as np 
import math

import tensorflow.keras.backend as K

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
        g_model = tg_model.g_model
        scale_factors = np.zeros(len(tg_model.layer_names))

        # For each synapse population
        for i, layer_name in enumerate(tg_model.layer_names):
            neurons = g_model.neuron_populations[layer_name + '_nrn']

            # For each sample
            for x in self.data:

                # Before simulation
                for ln in tg_model.layer_names:
                    nrn = g_model.neuron_populations[ln + '_nrn']
                    nrn.vars['Vmem'].view[:] = 0.0
                    nrn.vars['Vmem_peak'].view[:] = 0.0
                    nrn.vars['nSpk'].view[:] = 0
                    g_model.push_state_to_device(ln + '_nrn')

                # === Poisson inputs
                nrn = g_model.neuron_populations['input_nrn']
                nrn.vars['rate'].view[:] = x.flatten()
                g_model.push_state_to_device('input_nrn')

                # # === IF inputs with constant current
                # nrn = g_model.neuron_populations['input_nrn']
                # nrn.vars['Vmem'].view[:] = 0.0
                # nrn.vars['Vmem_peak'].view[:] = 0.0
                # nrn.vars['nSpk'].view[:] = 0
                # g_model.push_state_to_device('input_nrn')
                # cs = g_model.current_sources['input_cs']
                # cs.vars['magnitude'].view[:] = x.flatten()
                # g_model.push_state_to_device('input_cs')

                # Run simulation
                for t in range(math.ceil(self.present_time / g_model.dT)):
                    g_model.step_time()

                    g_model.pull_var_from_device(neurons.name, 'Vmem_peak')
                    Vmem_peak = neurons.vars['Vmem_peak'].view
                    scale_factors[i] = np.max([scale_factors[i], Vmem_peak.max()])

            # Update this neuron population's threshold
            neurons.extra_global_params['Vthr'].view[:] = scale_factors[i]

            # # Update this synapse population's weights
            # synapses = g_model.synapse_populations[layer_name + '_syn']
            # synapses.vars['g'].view[:] /= scale_factors[i]
            # g_model.push_var_to_device(synapses.name, 'g')

            print('layer: ' + layer_name)
            print(scale_factors)
