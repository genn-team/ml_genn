.. py:currentmodule:: ml_genn

Callbacks and recording
=======================
In order to run custom logic mid-simulation including for recording state, 
mlGeNN has a callback system (very similar to https://keras.io/api/callbacks/).
Currently the evaluate_XXX methods of compiled models all take a list of callback
objects (or the names of default-constructable callbacks in the same style as neuron 
models etc) which defaults to a list containing a :class:`~callbacks.progress_bar.BatchProgressBar`
to show inference progress. However, you could additionally add a :class:`~callbacks.var_recorder.VarRecorder`
callback to a model (where `input` is a :class:`~population.Population` object with a
neuron model that has a state variable called `V`):

..  code-block:: python

    from ml_genn.callbacks import VarRecorder
    ...
    callbacks = ["batch_progress_bar", VarRecorder(input, "V", key="v_input")]
    metrics, cb_data = compiled_net.evaluate({input: testing_images * 0.01}, {output: testing_labels},
                                              callbacks=callbacks)

to record the population's `V` state variable over time. After the simulation has 
completed, you could then plot the membrane voltage of all neurons using matplotlib with:

..  code-block:: python

    import matplotlib.pyplot as plt
    ...
    plt.plot(cb_data["v_input"][0])
    plt.show()

Keys are used to uniquely identify recorded data produced by callbacks and can be any 
hashable type. If no key is provided, the integer index of the callback will be used 
e.g. in this case, the key of the VarRecorder would be 1. Spike recording is very 
similar, you just need to use the :class:`~callbacks.spike_recorder.SpikeRecorder` 
callback instead. The only caveat is that, as this uses `GeNN's spike recording system <https://github.com/genn-team/genn/pull/372>`_,
you need to set the `record_spikes=True` keyword argument on :class:`~population.Population`, 
:class:`~layer.InputLayer` or :class:`~layer.Layer` objects when you construct the model. For example:

..  code-block:: python

    input = InputLayer(IntegrateFireInput(v_thresh=5.0), 784, record_spikes=True)

Filtering
---------
When dealing with large models/datasets, recording everything uses a lot of 
memory and slows the simulation down significantly. You can address this by adding 
filtering kwargs to :class:`~callbacks.spike_recorder.SpikeRecorder` and 
:class:`~callbacks.var_recorder.VarRecorder` objects. Example filters let you
select which examples to record from:

..  code-block:: python

    SpikeRecorder(input, example_filter=1000)    # Only record from example 1000
    SpikeRecorder(input, example_filter=[1000, 1002]) # Only record from examples 1000 and 1002
    SpikeRecorder(input, example_filter=[True]*10) # Only record from the first 10 examples

Similarly, neuron filters let you select which neurons to record from:

..  code-block:: python

    SpikeRecorder(input, neuron_filter=1000)    # Only record from neuron 1000 in a 1D population
    SpikeRecorder(input, neuron_filter=[1000, 1002]) # Only record from neurons 1000 and 1002 in a 1D population
    SpikeRecorder(input, neuron_filter=[True]*10) # Only record from the first 10 neurons in a 1D population
    SpikeRecorder(input, neuron_filter=np.s_[0::2]) # Only record from every other neuron in a 1D population

Because, in networks such as convolution neural networks, populations can have 
multidimensional shapes this syntax also extends to multiple dimensions in the same w
ay as numpy arrays, for example:

..  code-block:: python

    SpikeRecorder(input, neuron_filter=([16, 20], [16, 20])     # Record neurons(16,16) and (20, 20) in 2D population
    SpikeRecorder(input, neuron_filter=np.index_exp[2:4,2:4])   # Record neurons (2,2), (2,3), (3,2) and (3,3) in 2D population