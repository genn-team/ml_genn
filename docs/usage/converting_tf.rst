Converting TF models
====================
The :mod:`ml_genn_tf` module provides functionality for converting ANNs implemented
using Keras and TensorFlow into mlGeNN SNNs.:

..  code-block:: python

    # Create suitable compiler for model
    converter = FewSpike(k=10, signed_input=True, 
                         norm_data=[norm_data])

    # Convert and compile ML GeNN model
    net, net_inputs, net_outputs, tf_layer_pops = converter.convert(tf_model)

    compiler = converter.create_compiler()
    compiled_net = compiler.compile(net, inputs=net_inputs, outputs=net_outputs)


Once models have been converted, they can then be modified like any other mlGeNN model
by using the ``tf_layer_pops `` dictionary to get the mlGeNN populations corresponding 
to each layer. For example, to enable spike recording on a 