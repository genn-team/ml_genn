import tensorflow as tf
import json
from pygenn import genn_model, genn_wrapper

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

    return model

tf_model = train_mnist()
print(tf_model.summary())
tf_weights = tf_model.get_weights()
# print(tf_weights[0].shape,tf_weights[1].shape)

def convert_model(tf_model):
    # Custom classes
    if_model = genn_model.create_custom_neuron_class(
        "if_model",
        param_names=["Vres","Vthr","Cm"],
        var_name_types=[("Vmem","scalar")],
        sim_code="""
        $(Vmem) += $(Isyn) * (DT / $(Cm));
        """,
        reset_code="""
        $(Vmem) = $(Vres); 
        """,
        threshold_condition_code="$(V) >= $(Vthr"
    )()

    syn_model = genn_model.create_custom_postsynaptic_class(
        "syn_model",
        apply_input_code="""
        $(Isyn) += $(inSyn);
        """
    )()

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
        "Vmem":-65
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