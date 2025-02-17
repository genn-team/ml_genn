Inference
=========
The inference functionality in mlGeNN allows you to run trained networks efficiently.
The :class:`.compilers.InferenceCompiler` provides the core functionality for running inference on
trained networks:

.. autoclass:: ml_genn.compilers.InferenceCompiler
    :noindex:
    
Here's a basic example:

.. code-block:: python

    from ml_genn.compilers import InferenceCompiler

    # Create compiler with metric
    compiler = InferenceCompiler(
        example_timesteps=500,
        metrics="mean_square_error")

    # Compile network
    compiled_net = compiler.compile(network)

    # Run evaluation
    with compiled_net:
        metrics, _ = compiled_net.evaluate(
            {input_layer: x_test},
            {output_layer: y_test})

        print(f"Error: {metrics['mean_square_error']}")


The :class:`.compilers.CompiledInferenceNetwork` objects produced by the
:class:`.compilers.InferenceCompiler` provide methods for evaluation on 
datasets specified as sequences (typically numpy arrays or lists of 
:class:`.utils.data.PreprocessedSpikes`: objects):

.. automethod:: ml_genn.compilers.CompiledInferenceNetwork.evaluate
    :noindex:

Alternatively, you can evaluate a model on a dataset iterator (such as a :

.. automethod:: ml_genn.compilers.CompiledInferenceNetwork.evaluate_batch_iter
    :noindex:

Finally, raw predictions (i.e. the output of your model's readouts) 
can be obtained on a dataset:

.. automethod:: ml_genn.compilers.CompiledInferenceNetwork.predict
    :noindex:
    
Like in Keras, additional logic such as checkpointing and recording of 
state variables can be added to any of these standard inference loops using callbacks 
as described in the :ref:`section-callbacks-recording` section.

