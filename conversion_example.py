import tensorflow as tf
import json
from pygenn import genn_model, genn_wrapper
import numpy as np
import random

def train_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu', use_bias=False),
    tf.keras.layers.Dense(10, activation='softmax',use_bias=False)
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test[:1000], y_test[:1000])

    return model, x_train, y_train, x_test, y_test

tf_model, x_train, y_train, x_test, y_test = train_mnist()
print(tf_model.summary())
tf_weights = tf_model.get_weights()

print(tf_weights[0].shape,tf_weights[1].shape)
print(tf_model.layers)

def convert_model(tf_model):
    # Hyperparameters
    Vres = -60.0
    Vthr = -55.0
    Cm = 1.0
    timestep = 1.0

    # Custom classes
    if_model = genn_model.create_custom_neuron_class(
        "if_model",
        param_names=["Vres","Vthr","Cm"],
        var_name_types=[("Vmem","scalar"),("SpikeNumber","unsigned int")],
        sim_code="""
        $(Vmem) += $(Isyn)*(DT / $(Cm));
        //printf("Vmem: %f, Isyn: %f, SpikeNumber: %d", $(Vmem),$(Isyn),$(SpikeNumber));
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

    # Fetch tf_model details
    n_layers = len(tf_model.layers)
    tf_weights = tf_model.get_weights()

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

    # Define model and populations
    g_model = genn_model.GeNNModel("float","mnist")
    neuron_pops = {}
    syn_pops = {}

    for i in range(1,n_layers): # 1,2 - synapses
        if i == 1:
            # Presynaptic neuron
            neuron_pops["if"+str(i-1)] = g_model.add_neuron_population(
                "if"+str(i-1),tf_weights[i-1].shape[0],if_model,if_params,if_init
            )

        # Postsynaptic neuron
        neuron_pops["if"+str(i)] = g_model.add_neuron_population(
            "if"+str(i),tf_weights[i-1].shape[1],if_model,if_params,if_init
        )

        # Synapse
        syn_pops["syn"+str(i-1)+str(i)] = g_model.add_synapse_population(
            "syn"+str(i-1)+str(i),"DENSE_INDIVIDUALG",genn_wrapper.NO_DELAY,
            neuron_pops["if"+str(i-1)], neuron_pops["if"+str(i)],
            "StaticPulse",{},{'g':tf_weights[i-1].reshape(-1)},{},{},
            "DeltaCurr",{},{}
        )
    
    current_source = g_model.add_current_source("cs",cs_model,"if0",{},cs_init)

    g_model.dT = timestep
    g_model.build()
    g_model.load()

    return g_model, neuron_pops, syn_pops, current_source

# Create models
g_model, neuron_pops, syn_pops, current_source = convert_model(tf_model)
print(g_model)
print(neuron_pops)
print(syn_pops)
print(current_source)


# Evaluate on test set
X = (x_test*255.0).reshape(10000,-1)
y = y_test.reshape(10000)

n_examples = 1000
single_example_time = 350.
runtime = n_examples * single_example_time
i = -1
preds = []

n_correct = 0
while g_model.t <= runtime:
    if g_model.t >= single_example_time*(i+1):
        # After example i -1,0,1,2,..
        g_model.pull_var_from_device("if2",'SpikeNumber')
        SpikeNumber_view = neuron_pops["if2"].vars["SpikeNumber"].view
        print("Example {}, Time {}, True label {}, Pred label {}".format(i,g_model.t,y[i],np.argmax(SpikeNumber_view)))
        n_correct += (np.argmax(SpikeNumber_view)==y[i])
        i += 1
        # Before example i 0,1,2,3,..
        # g_model.reinitialise()
        for j in range(len(neuron_pops)):
            neuron_pops["if"+str(j)].vars["SpikeNumber"].view[:] = 0
            neuron_view = neuron_pops["if"+str(j)].vars["Vmem"].view[:] = random.uniform(-60.,-55.)
            g_model.push_state_to_device("if"+str(j))
        
        magnitude_view = current_source.vars['magnitude'].view[:] = X[i] / 100.
        g_model.push_var_to_device("cs",'magnitude')

    g_model.step_time()

accuracy = (n_correct / n_examples) * 100.
print(accuracy)
