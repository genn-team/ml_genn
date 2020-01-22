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
        genn_model = tg_model.genn_model

        n_syn_pops = len(tg_model.synapse_pops)
        scale_factors = np.zeros(n_syn_pops)

        tg_model.genn_w_norm = []

        # For each synapse population
        for syn_pop in range(n_syn_pops):



            # CONV2D LAYERS ONLY
            #if not syn_pop in [0, 1]:
            #    continue



            # For each sample
            for x in self.data:

                # Before simulation
                for i, npop in enumerate(tg_model.neuron_pops):
                    npop.vars['Vmem'].view[:] = 0.0
                    npop.vars['Vmem_peak'].view[:] = 0.0
                    npop.vars['nSpk'].view[:] = 0
                    genn_model.push_state_to_device('if' + str(i))


                # TODO: INPUT RATE ENCODING
                # FOR NOW, USE CONSTANT CURRENT INJECTION EQUAL TO INPUT MAGNITUDE
                tg_model.current_source.vars['magnitude'].view[:] = x.flatten()
                genn_model.push_var_to_device('cs', 'magnitude')


                # Run simulation
                for t in range(math.ceil(self.present_time / genn_model.dT)):
                    genn_model.step_time()

                    genn_model.pull_var_from_device('if' + str(syn_pop+1), 'Vmem_peak')
                    Vmem_peak = tg_model.neuron_pops[syn_pop+1].vars['Vmem_peak'].view
                    scale_factors[syn_pop] = np.max([scale_factors[syn_pop], Vmem_peak.max()])


            print('syn_pop = ' + str(syn_pop))
            print(scale_factors)




            print('genn_w_vals shape: ')
            print(tg_model.genn_w_vals[syn_pop].shape)



            # Update this synapse population's weights
            genn_w = tg_model.genn_w_vals[syn_pop] / scale_factors[syn_pop]
            tg_model.genn_w_norm.append(genn_w)



            #print('  TG w shape:  ' + str(genn_w.shape))
            #print('GeNN w shape:  ' + str(tg_model.synapse_pops[syn_pop].vars['g'].view.shape))



            # ========= WHY IS GENN_W_VALS SIZE DIFFERENT TO GENN G WEIGHTS???
            # BECAUSE RAGGED ARRAY CONVERSION?



            import matplotlib.pyplot as plt

            # if syn_pop == 0:
            #     plt.figure()
            #     plt.plot(tg_model.genn_w_vals[syn_pop])
            #     plt.figure()
            #     genn_model.pull_var_from_device('syn' + str(syn_pop) + str(syn_pop+1), 'g')
            #     plt.plot(tg_model.synapse_pops[syn_pop].vars['g'].view)
            #     plt.show()


            # if syn_pop == 0:
            #     ic = 1
            #     oc = 16

            #     test = np.full((tg_model.genn_n_neurons[syn_pop], tg_model.genn_n_neurons[syn_pop + 1]), np.nan)
            #     inds = tg_model.genn_w_inds[syn_pop]
            #     vals = tg_model.genn_w_vals[syn_pop]

            #     for i in range(len(vals)):
            #         test[inds[0][i], inds[1][i]] = vals[i]

            #     #plt.matshow(test)
            #     plt.matshow(test[0::ic, 0::oc])
            #     plt.show()



            # ======== SHOULDNT REALLY NEED TO DO THIS
            #genn_model.pull_var_from_device('syn' + str(syn_pop) + str(syn_pop+1), 'g')

            # ========= REPLACE WITH GENN WEIGHT UPDATE METHOD
            #tg_model.synapse_pops[syn_pop].vars['g'].view[:] = genn_w
            #genn_model.push_var_to_device('syn' + str(syn_pop) + str(syn_pop+1), 'g')




            # Update this synapse population's weights
            genn_model.pull_var_from_device('syn' + str(syn_pop) + str(syn_pop+1), 'g')
            tg_model.synapse_pops[syn_pop].vars['g'].view[:] /= scale_factors[syn_pop]
            genn_model.push_var_to_device('syn' + str(syn_pop) + str(syn_pop+1), 'g')


            # # Update this neuron population's thresholds
            # genn_model.pull_var_from_device('if' + str(syn_pop+1), 'Vthr')
            # tg_model.neuron_pops[syn_pop+1].vars['Vthr'].view[:] = scale_factors[syn_pop]
            # genn_model.push_var_to_device('if' + str(syn_pop+1), 'Vthr')
