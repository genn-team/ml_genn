.. py:currentmodule:: ml_genn

.. _section-metrics:

Metrics
=======
mlGeNN can calculate different types of metrics when evaluating datasets. 
By default, the ``SparseCategoricalAccuracy`` metric is calculated for all output 
neurons but this can be overridden. For example, in a regression task you could 
instead use:

..  code-block:: python

    metrics = compiled_net.evaluate({input: x}, {output: y}, "mean_square_error")

to calculate a mean square error. Finally, if a model has multiple outputs, 
different metrics can be calculated for each one by providing a dictionary of metrics:

..  code-block:: python

    metrics = compiled_net.evaluate({input: x}, {output: y, output_2: y2},
		                    {output: "mean_square_error", output_2: "sparse_categorical_accuracy"})
    print(f"Error = {metrics[output].result}")
    print(f"Accuracy = {100 * metrics[output_2].result}%")
