Converting TF models
====================
The :mod:`ml_genn_tf` module provides functionality for converting ANNs implemented
using Keras and TensorFlow into mlGeNN SNNs.
For example if we define the following simple CNN in Keras:

..  code-block:: python

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Flatten

    tf_model = Sequential([
       Conv2D(16, 3, padding='same', activation='relu',
              use_bias=False, input_shape=(32, 32, 3)),
       AveragePooling2D(2),
       Flatten(),
       Dense(10, activation='relu', use_bias=False)])

After training the weights with TensorFlow, the model can be converted to a mlGeNN SNN. For example, to use the few-spike [Stockl2021]_ method, you could do the following:

..  code-block:: python

    # Create suitable compiler for model
    converter = FewSpike(k=10, signed_input=True, 
                         norm_data=[norm_data])

    # Convert and compile ML GeNN model
    net, net_inputs, net_outputs, tf_layer_pops = converter.convert(tf_model)


Once tensorflow models have been converted, they can then be modified like any other mlGeNN model by using the ``tf_layer_pops`` dictionary returned by the conveter's convert method to get the mlGeNN :class:`ml_genn.Population` corresponding to each layer. 
For example, to enable spike recording on the first layer you would do this:

..  code-block:: python

    tf_layer_pops[tf_model.get_layer(0)].record_spikes = True

Finally, the converter can create a suitable compiler and this can be used to compile the mlGeNN network:

..  code-block:: python
    
    compiler = converter.create_compiler()
    compiled_net = compiler.compile(net, inputs=net_inputs, outputs=net_outputs)
