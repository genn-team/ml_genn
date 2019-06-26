import numpy as np
import tensorflow as tf 
import random
import math
import sys

from pygenn import genn_model, genn_wrapper, genn_groups

Vres = -60.0
Vthr = -55.0
Cm = 0.4
timestep = 1.0
single_example_time = 350.0
n_inputs = 784

def train_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_normed, x_test_normed = x_train / 255.0, x_test / 255.0

    x_train_normed, x_test_normed = x_train_normed.reshape((-1,28,28,1)), x_test_normed.reshape((-1,28,28,1))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,5,activation='relu',use_bias=False,input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(8,5,activation='relu',use_bias=False),
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(128, activation='relu', use_bias=False),
        tf.keras.layers.Dense(64, activation='relu', use_bias=False),
        tf.keras.layers.Dense(10, activation='softmax',use_bias=False)
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train_normed[:10000], y_train[:10000], epochs=1)

    model.evaluate(x_test_normed[:1000], y_test[:1000])

    return model, x_train, y_train, x_test, y_test

tf_model, x_train, y_train, x_test, y_test = train_mnist()
print(tf_model.summary())

def create_conv_weight_matrices(tf_model):
    tf_weights = tf_model.get_weights()
    tf_layers = tf_model.layers

    g_model_weights = []
    n_units = [n_inputs]
    j=1
    for i, layer in enumerate(tf_layers):
        if not isinstance(layer,tf.keras.layers.Flatten):
            n_units.append(np.prod(layer.output_shape[1:]))
            syn_weights = np.zeros((n_units[j-1],n_units[j]))

            if isinstance(layer,tf.keras.layers.Conv2D):
                kw,kh = layer.kernel_size # 2,2
                sw,sh = layer.strides # 1,1
                ih,iw,ic = layer.input_shape[1:] # 3,3,2
                oh,ow,oc = layer.output_shape[1:] # 2,2,2
    
                for n in range(int(n_units[j])): # output unit/conv number
                    for k in range(kh): # kernel height
                        syn_weights[((n//oc)%ow)*ic*sw + ((n//oc)//ow)*ic*iw*sh + k*ic*iw:
                                    ((n//oc)%ow)*ic*sw + ((n//oc)//ow)*ic*iw*sh + k*ic*iw + kw*ic,
                                    n] = tf_weights[j-1][k,:,:,n%oc].reshape((-1))
                        
            elif isinstance(layer,tf.keras.layers.Dense):
                syn_weights = tf_weights[j-1]
            g_model_weights.append(syn_weights)
            j += 1
    
    return g_model_weights, n_units

g_model_weights, n_units = create_conv_weight_matrices(tf_model)

def convert_model(tf_model):
    # tf_model parameters
    n_layers = len(tf_model.layers)

    # create custom GeNN classes
    if_model = genn_model.create_custom_neuron_class(
        "if_model",
        param_names=["Vres","Vthr","Cm"],
        var_name_types=[("Vmem","scalar"),("SpikeNumber","unsigned int")],
        sim_code="""
        $(Vmem) += $(Isyn)*(DT / $(Cm));
        """,
        reset_code="""
        $(Vmem) = $(Vres); 
        $(SpikeNumber) += 1;
        """,
        threshold_condition_code="$(Vmem) >= $(Vthr)"
    )

    cs_model = genn_model.create_custom_current_source_class(
        "cs_model",
        var_name_types=[("magnitude","scalar")],
        injection_code="""
        $(injectCurrent, $(magnitude));
        """
    )

    # Params and init
    if_params = {
        "Vres":Vres,
        "Vthr":Vthr,
        "Cm":Cm
    }

    if_init = {
        "Vmem":genn_model.init_var("Uniform", {"min": Vres, "max": Vthr}),
        "SpikeNumber":0
    }
    
    cs_init = {"magnitude":10.0}

    # Fetch augmented weight matrices
    g_weights, n_units = create_conv_weight_matrices(tf_model)
    gw_inds = [np.nonzero(gw) for gw in g_weights]
    gw_vals = [g_weights[i][gw_inds[i]].reshape(-1) for i in range(len(g_weights))]

    # Model conversion
    g_model = genn_model.GeNNModel("float","g_model")

    neuron_pops = []
    syn_pops = []
    for i in range(1,n_layers):
        if i==1:
            # presynaptic neuron
            neuron_pops.append(g_model.add_neuron_population(
                "if"+str(i-1),n_units[i-1],if_model,if_params,if_init
            ))
        
        # Postsynaptic neuron
        neuron_pops.append(g_model.add_neuron_population(
            "if"+str(i),n_units[i],if_model,if_params,if_init
        ))

        # Synapse
        syn_pops.append(g_model.add_synapse_population(
            "syn"+str(i-1)+str(i),"SPARSE_INDIVIDUALG",genn_wrapper.NO_DELAY,
            neuron_pops[i-1],neuron_pops[i],
            "StaticPulse",{},{'g':gw_vals[i-1]},{},{},
            "DeltaCurr",{},{}))

        syn_pops[-1].set_sparse_connections(gw_inds[i-1][0],gw_inds[i-1][1])

    current_source = g_model.add_current_source("cs",cs_model,"if0",{},cs_init)

    g_model.dT = timestep 
    g_model.build()
    g_model.load()

    return g_model, neuron_pops, current_source

g_model, neuron_pops, current_source = convert_model(tf_model)  

def evaluate(X, y=None):
    n_examples = len(X)
    X = X.reshape(n_examples,-1)
    y = y.reshape(n_examples)

    n = len(neuron_pops)

    n_correct = 0

    for i in range(n_examples):
        # Before simulation
        for j, npop in enumerate(neuron_pops):
                npop.vars["SpikeNumber"].view[:] = 0
                npop.vars["Vmem"].view[:] = random.uniform(Vres,Vthr)
                g_model.push_state_to_device("if"+str(j))
            
        current_source.vars['magnitude'].view[:] = X[i] / 100.
        g_model.push_var_to_device("cs",'magnitude')

        # Run simulation
        for _ in range(math.ceil(single_example_time/timestep)):
            g_model.step_time()

        # After simulation
        g_model.pull_var_from_device("if"+str(n-1),'SpikeNumber')
        SpikeNumber_view = neuron_pops[-1].vars["SpikeNumber"].view

        n_correct += (np.argmax(SpikeNumber_view)==y[i])

    accuracy = (n_correct / n_examples) * 100.
    
    return accuracy

print(evaluate(x_test[:100],y_test[:100]))