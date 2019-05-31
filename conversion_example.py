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
# print(tf_weights[0].shape,tf_weights[1].shape)

def convert_model(tf_model):
    # Custom classes
    if_model = genn_model.create_custom_neuron_class(
        "if_model",
        param_names=["Vres","Vthr","Cm"],
        var_name_types=[("Vmem","scalar"),("SpikeNumber","unsigned int")],
        sim_code="""
        $(Vmem) += $(Isyn) * (DT / $(Cm));
        """,
        reset_code="""
        $(Vmem) = $(Vres); 
        $(SpikeNumber) += 1;
        """,
        threshold_condition_code="$(Vmem) >= $(Vthr)"
    )

    syn_model = genn_model.create_custom_postsynaptic_class(
        "syn_model",
        apply_input_code="""
        $(Isyn) += $(inSyn);
        """
    )

    # Fetch tf_model details
    n_layers = len(tf_model.layers)
    tf_weights = tf_model.get_weights()

    # print(n_layers)
    # print(tf_weights[0].shape,tf_weights[0].reshape(-1,1).shape)

    # Params and init
    if_params = {
        "Vres":-65,
        "Vthr":-52,
        "Cm":0.001
    }
    if_init = {
        "Vmem":-65,
        "SpikeNumber":0
    }

    # Define model and populations
    g_model = genn_model.GeNNModel("float","mnist")
    neuron_pops = {}
    syn_pops = {}
    
    for i in range(1,n_layers):
        neuron_pops["layer"+str(i)] = g_model.add_neuron_population(
            "if"+str(i),tf_weights[i-1].shape[1],
            if_model,if_params,if_init)
        
        if (i>1):
            syn_pops["syn"+str(i)+str(i-1)] = g_model.add_synapse_population(
                "syn"+str(i)+str(i-1),"DENSE_INDIVIDUALG",genn_wrapper.NO_DELAY,
                neuron_pops["layer"+str(i-1)], neuron_pops["layer"+str(i)],
                "StaticPulse",{},{"g":list(tf_weights[i-1].reshape(-1))},{},{},
                syn_model,{},{}
            )

    return g_model, neuron_pops, syn_pops

g_model, neuron_pops, syn_pops = convert_model(tf_model)

def convert_data(g_model,neuron_pops,syn_pops,tf_model):
    tf_weights = tf_model.get_weights()

    poisson_model = genn_model.create_custom_neuron_class(
        "poisson_model",
        var_name_types=[("timeStepToSpike", "scalar"),("isi","scalar")],
        sim_code="""
        if($(timeStepToSpike) > 0){
            $(timeStepToSpike) -= 1.0;
        }
        """,
        reset_code="""
        $(timeStepToSpike) += 1.0 / $(isi);
        """,
        threshold_condition_code="$(timeStepToSpike) <= 0.0"
    )

    syn_model = genn_model.create_custom_postsynaptic_class(
        "syn_model",
        apply_input_code="""
        $(Isyn) += $(inSyn);
        """
    )

    poisson_init = {
    "timeStepToSpike": 0.8,
    "isi": 0.0
    }

    poisson_pop = g_model.add_neuron_population(
        "poisson_pop",tf_weights[0].shape[0],poisson_model,{},poisson_init
    )

    # g_model.add_current_source(
    #     "current_source","DC","poisson_pop",{"amp":10.0},{}
    # )

    input_pop = g_model.add_synapse_population("input_pop","DENSE_INDIVIDUALG",genn_wrapper.NO_DELAY,
        poisson_pop,neuron_pops["layer1"],
        "StaticPulse",{},{"g":list(tf_weights[0].reshape(-1))},{},{},
        syn_model,{},{}
    )

    return g_model, input_pop, poisson_pop

g_model, input_pop, poisson_pop = convert_data(g_model,neuron_pops,syn_pops,tf_model)

print(input_pop)

# Evaluate on training set
X = (x_train*255.0).reshape(60000,-1)
y = y_train.reshape(60000)
g_model.dT = 1.0

g_model.build()
g_model.load()

n_examples = X.shape[0]
single_example_time = 350
runtime = n_examples * single_example_time
i = 0
preds = []
isi_view = poisson_pop.vars['isi'].view
while g_model.t < runtime:
    if g_model.t >= single_example_time*i:
        i+=1
        isi = np.array(1.0/(X[i] + 1)) * 1000
        poisson_pop.set_var("isi",isi)
        poisson_pop.set_var("timeStepToSpike",0.0)
        g_model.push_var_to_device("poisson_pop","isi")
        g_model.push_var_to_device("poisson_pop","timeStepToSpike")
        print(isi_view)

    g_model.step_time()
    g_model.pull_current_spikes_from_device("poisson_pop")
    print(poisson_pop.current_spikes)