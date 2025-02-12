Inference
=========

The inference functionality in mlGeNN allows you to run trained networks efficiently.
This section covers how to use the inference compiler and its compiled networks.

Creating an Inference Compiler
----------------------------
The ``InferenceCompiler`` provides the core functionality for running inference on
trained networks. Here's a basic example:

.. code-block:: python

    from ml_genn.compilers import InferenceCompiler

    # Create compiler
    compiler = InferenceCompiler(example_timesteps=500)

    # Compile network
    compiled_net = compiler.compile(network, inputs=net_inputs, outputs=net_outputs)

Key Methods
----------

predict()
~~~~~~~~
The ``predict()`` method runs inference on input data and returns predictions:

.. code-block:: python

    with compiled_net:
        predictions = compiled_net.predict({input_layer: input_data})

Parameters:
    * ``inputs`` - Dictionary mapping input layers to input data
    * ``batch_size`` - Optional batch size for processing (default: None)
    * ``reset_time`` - Whether to reset time between samples (default: True)

Returns:
    Dictionary mapping output layers to their predictions

evaluate()
~~~~~~~~~
The ``evaluate()`` method runs inference and calculates metrics:

.. code-block:: python

    with compiled_net:
        metrics, callback_data = compiled_net.evaluate(
            {input_layer: x_test},
            {output_layer: y_test}
        )

Parameters:
    * ``inputs`` - Dictionary mapping input layers to input data
    * ``targets`` - Dictionary mapping output layers to target data
    * ``batch_size`` - Optional batch size for processing (default: None)
    * ``reset_time`` - Whether to reset time between samples (default: True)

Returns:
    * ``metrics`` - Dictionary containing evaluation metrics
    * ``callback_data`` - Additional data collected during evaluation

Configuration Options
-------------------
The ``InferenceCompiler`` accepts several configuration parameters:

.. code-block:: python

    compiler = InferenceCompiler(
        example_timesteps=500,    # Number of timesteps per example
        batch_size=32,           # Default batch size
        reset_time=True,         # Reset time between samples
        kernel_profiling=False   # Enable kernel profiling
    )

Advanced Features
---------------

Custom Metrics
~~~~~~~~~~~~
You can define custom metrics for evaluation:

.. code-block:: python

    def custom_accuracy(output_data, target_data):
        # Custom metric calculation
        return accuracy_value

    compiler = InferenceCompiler(metrics={'accuracy': custom_accuracy})

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

Best Practices
------------

Memory Management
~~~~~~~~~~~~~~~
* Use appropriate batch sizes to manage memory usage
* Clear GPU memory between large inference runs
* Consider using ``reset_time=False`` for continuous inference

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~
* Enable kernel profiling to identify bottlenecks
* Adjust ``example_timesteps`` based on network dynamics
* Use appropriate batch sizes for your GPU memory

Common Issues
-----------

GPU Memory Errors
~~~~~~~~~~~~~~~
If encountering GPU memory errors:

* Reduce batch size
* Clear GPU memory between runs
* Check for memory leaks in custom callbacks

Performance Issues
~~~~~~~~~~~~~~~
If experiencing slow inference:

* Profile kernels to identify bottlenecks
* Optimize network architecture
* Adjust simulation timesteps

Complete Example
--------------
Here's a complete example showing typical usage:

.. code-block:: python

    from ml_genn.compilers import InferenceCompiler

    # Define custom metric
    def spike_rate(output_data, target_data):
        return np.mean(output_data > 0)

    # Create compiler with custom metric
    compiler = InferenceCompiler(
        example_timesteps=500,
        metrics={'spike_rate': spike_rate}
    )

    # Compile network
    compiled_net = compiler.compile(network)

    # Run evaluation
    with compiled_net:
        metrics, _ = compiled_net.evaluate(
            {input_layer: x_test},
            {output_layer: y_test}
        )
        print(f"Spike rate: {metrics['spike_rate']}")