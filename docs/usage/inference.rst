Inference
=========

The inference functionality in mlGeNN allows you to run trained networks efficiently.
This section covers how to use the inference compiler and its compiled networks.

Creating an Inference Compiler
----------------------------
The :class:`.compilers.InferenceCompiler` provides the core functionality for running inference on
trained networks. Here's a basic example:

.. code-block:: python

    from ml_genn.compilers import InferenceCompiler

    # Create compiler
    compiler = InferenceCompiler(example_timesteps=500)

    # Compile network
    compiled_net = compiler.compile(network)

Key Methods
----------

predict()
~~~~~~~~
.. automethod:: ml_genn.compilers.InferenceCompiler.predict
    :noindex:

evaluate()
~~~~~~~~~
.. automethod:: ml_genn.compilers.InferenceCompiler.evaluate
    :noindex:

Configuration Options
-------------------
The :class:`.compilers.InferenceCompiler` accepts several configuration parameters:

.. code-block:: python

    compiler = InferenceCompiler(
        example_timesteps=500,    # Number of timesteps per example
        batch_size=32,           # Default batch size
        reset_time=True,         # Reset time between samples
        kernel_profiling=False   # Enable kernel profiling
    )

Advanced Features
---------------

Using Callbacks
~~~~~~~~~~~~~
The compiler supports callbacks for monitoring inference:

.. code-block:: python

    class CustomCallback:
        def on_batch_begin(self, batch):
            pass
        
        def on_batch_end(self, batch):
            pass

    compiler = InferenceCompiler(callbacks=[CustomCallback()])

Complete Example
--------------
Here's a complete example showing typical usage:

.. code-block:: python

    from ml_genn.compilers import InferenceCompiler

    # Create compiler with metric
    compiler = InferenceCompiler(
        example_timesteps=500,
        metrics="mean_square_error"
    )

    # Compile network
    compiled_net = compiler.compile(network)

    # Run evaluation
    with compiled_net:
        metrics, _ = compiled_net.evaluate(
            {input_layer: x_test},
            {output_layer: y_test}
        )
        print(f"Error: {metrics['mean_square_error']}")