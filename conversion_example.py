import tensorflow as tf
import json
from pygenn import genn_model, genn_wrapper
import numpy as np

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

    model.fit(x_train, y_train, epochs=2)

    model.evaluate(x_test, y_test)

    return model, x_train, y_train

tf_model, x_train, y_train = train_mnist()
print(tf_model.summary())
tf_weights = tf_model.get_weights()

def convert_model(tf_model):
    # Hyperparameters
    Vres = -60.0
    Vthr = -50.0
    Cm = 0.01
    timestep = 1.0
    scale = 0.2

    # Custom classes
    poisson_model = genn_model.create_custom_neuron_class(
        "poisson_model",
        var_name_types=[("timeStepToSpike", "scalar"),("isi","scalar")],
        sim_code="""
        $(Isyn);
        if($(timeStepToSpike) > 0){
            $(timeStepToSpike) -= 1.0;
        }
        """,
        reset_code="""
        $(timeStepToSpike) += $(isi);
        """,
        threshold_condition_code="$(timeStepToSpike) <= 0.0"
    )

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

    # Fetch tf_model details
    n_layers = len(tf_model.layers)
    tf_weights = tf_model.get_weights()

    # Params and init
    poisson_init = {
    "timeStepToSpike": 0.0,
    "isi": 0.0
    }

    if_params = {
        "Vres":Vres,
        "Vthr":Vthr,
        "Cm":Cm
    }
    if_init = {
        "Vmem":genn_model.init_var("Uniform", {"min": Vres, "max": Vthr}),
        "SpikeNumber":0
    }

    # Define model and populations
    g_model = genn_model.GeNNModel("float","mnist")
    neuron_pops = {}
    syn_pops = {}
    
    poisson_pop = g_model.add_neuron_population(
        "poisson_pop",tf_weights[0].shape[0],poisson_model,{},poisson_init
    )

    for i in range(1,n_layers):
        neuron_pops["if"+str(i)] = g_model.add_neuron_population(
            "if"+str(i),tf_weights[i-1].shape[1],
            if_model,if_params,if_init)
        
        if (i>1):
            syn_pops["syn"+str(i-1)+str(i)] = g_model.add_synapse_population(
                "syn"+str(i-1)+str(i),"DENSE_INDIVIDUALG",genn_wrapper.NO_DELAY,
                neuron_pops["if"+str(i-1)], neuron_pops["if"+str(i)],
                "StaticPulse",{},{"g":tf_weights[i-1].reshape(-1)},{},{},
                "DeltaCurr",{},{}
            )
    
    input_pop = g_model.add_synapse_population("input_pop","DENSE_INDIVIDUALG",genn_wrapper.NO_DELAY,
        poisson_pop,neuron_pops["if1"],
        "StaticPulse",{},{"g":tf_weights[0].reshape(-1)},{},{},
        "DeltaCurr",{},{}
    )

    g_model.add_current_source("current_source","DC","poisson_pop",{"amp":10.0},{})

    g_model.dT = timestep
    g_model.build()
    g_model.load()

    return g_model, neuron_pops, syn_pops, input_pop, poisson_pop

# Create models
g_model, neuron_pops, syn_pops, input_pop, poisson_pop = convert_model(tf_model)
print(g_model)
print(neuron_pops)
print(syn_pops)
print(input_pop)
print(poisson_pop)

# Evaluate on training set
X = (x_train*255.0).reshape(60000,-1)
y = y_train.reshape(60000)

n_examples = X.shape[0]/60
single_example_time = 350
runtime = n_examples * single_example_time
i = 0
preds = []
while g_model.t < runtime:
    if g_model.t >= single_example_time*i:
        print("Example: {}, Time: {}, Label: {}".format(i+1,g_model.t,y[i]))
        i+=1
        isi = 400.0/(X[i] + 1)
        isi_view = poisson_pop.vars['isi'].view
        timeStepToSpike_view = poisson_pop.vars['timeStepToSpike'].view
        isi_view[:] = isi
        timeStepToSpike_view[:] = 0.0
        g_model.push_state_to_device("input_pop")

    g_model.step_time()
    # g_model.pull_current_spikes_from_device("if1")
    # spnum = neuron_pops['if1'].current_spikes
    # print(spnum)