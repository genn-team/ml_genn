.. py:currentmodule:: ml_genn

.. _section-callbacks-recording:

Callbacks and recording
=======================
In order to run custom logic mid-simulation including for recording state, 
mlGeNN has a callback system (very similar to https://keras.io/api/callbacks/).
Currently the ``train`` and ``evaluate`` methods of compiled models all take a list of callback
objects (or the names of default-constructable callbacks in the same style as neuron 
models etc) which defaults to a list containing a :class:`~callbacks.progress_bar.BatchProgressBar`
to show training or inference progress.
As well as the callbacks mlGeNN uses to expose spike and variable recording functionality which will be
described in more depth in the following sections, mlGeNN provides the :class:`~callbacks.Checkpoint` callback 
for checkpointing weights and other learnable parameters throughout training and the 
:class:`~callbacks.OptimiserParamSchedule` callback for implementing learning rate schedules.

Recording
---------
It is often very useful to record spike trains and the value of state variables throughout simulation.
In mlGeNN, this functionality is implemented using callbacks.

Spikes
^^^^^^
Because spike recording uses `GeNN's spike recording system <https://github.com/genn-team/genn/pull/372>`_,
you need to set the ``record_spikes=True`` keyword argument on :class:`~population.Population`, 
:class:`~layer.InputLayer` or :class:`~layer.Layer` objects you wish to record spikes from when you construct the model. 
For example:

..  code-block:: python

    input = InputLayer(IntegrateFireInput(v_thresh=5.0), 784, record_spikes=True)

Then you can add :class:`~callbacks.var_recorder.SpikeRecorder` callbacks to a model to record spikes:

..  code-block:: python

    from ml_genn.callbacks import VarRecorder
    ...
    callbacks = ["batch_progress_bar", SpikeRecorder(input, key="spikes_input")]
    metrics, cb_data = compiled_net.evaluate({input: testing_images * 0.01},
		                             {output: testing_labels},
                                             callbacks=callbacks)

The ``key`` argument is used to uniquely identify data produced by callbacks in the ``cb_data``  dictionary
returned by ``evaluate`` and can be any hashable type. If no key is provided, the integer index of the 
callback will be used e.g. in this case, the key of the SpikeRecorder would be 1. For example, the following code-block
produces a raster plot of all the spikes emitted by all neurons during the fifth example using matplotlib:

..  code-block:: python
    
    import matplotlib.pyplot as plt
    ...
    spike_times = cb_data["spikes_input"][0][4],
    spike_ids = cb_data["spikes_input"][1][4]
    
    plt.scatter(spike_times, spike_ids)
    plt.show()


Variables
^^^^^^^^^
You can add :class:`~callbacks.var_recorder.VarRecorder` callbacks to a model to record neuron model state variables. 
For example, to record a state variable called `v` from a  :class:`~population.Population` object `input`:

..  code-block:: python

    from ml_genn.callbacks import VarRecorder
    ...
    callbacks = ["batch_progress_bar", VarRecorder(input, "v", key="v_input")]
    metrics, cb_data = compiled_net.evaluate({input: testing_images * 0.01},
                                             {output: testing_labels},
                                             callbacks=callbacks)

to record the population's `V` state variable over time. After the simulation has 
completed, you could then plot the membrane voltage of all neurons during the first example using matplotlib with:

..  code-block:: python

    import matplotlib.pyplot as plt
    ...
    plt.plot(cb_data["v_input"][0])
    plt.show()

You can also use a :class:`~callbacks.conn_var_recorder.ConnVarRecorder` callback to record connection state variables.
By convention, connection weights are held in a variable called `g` and delays in a variable called `d`.
Other variables may be available depending on the compiler being used (e.g. if you are training with EventProp, weight gradients are held in a variable called `Gradient` ).


Filtering
^^^^^^^^^
When dealing with large models/datasets, recording everything uses a lot of 
memory and slows the simulation down significantly. You can address this by adding 
filtering kwargs to :class:`~callbacks.spike_recorder.SpikeRecorder`, 
:class:`~callbacks.var_recorder.VarRecorder` and :class:`~callbacks.conn_var_recorder.ConnVarRecorder` objects. 
Example filters let you select which examples to record from:

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

Because, in networks such as convolutional neural networks, populations can have 
multidimensional shapes this syntax also extends to multiple dimensions in the same way
as numpy arrays, for example:

..  code-block:: python

    SpikeRecorder(input, neuron_filter=([16, 20], [16, 20])     # Record neurons(16,16) and (20, 20) in 2D population
    SpikeRecorder(input, neuron_filter=np.index_exp[2:4,2:4])   # Record neurons (2,2), (2,3), (3,2) and (3,3) in 2D population

When using :class:`~callbacks.conn_var_recorder.ConnVarRecorder`, seperate neuron filters can be provided for the connection's source and target population.
For example:

..  code-block:: python

    ConnVarRecorder(in_to_hid, "g", src_neuron_filter=10, trg_neuron_filter=10)
    
only records from synapse connecting neuron 10 in the source population to neuron 10 in the target population.

    
Custom callbacks
----------------
Beyond the built in callbacks, the callback system is intended to be the easiest way for users to
plug their own functionality into the training and inference workflows provided by mlGeNN.
Implementing your own callback is as easy as deriving a new class from :class:`~callbacks.Callback`.
Callbacks can implement any of the following methods which allow them to be triggered at any point in the simulation:

* ``on_test_begin(self)``: called at start of inference
* ``on_test_end(self, metrics)``: called at end of inference with metrics (see :ref:`section-metrics`) calculated from test set
* ``on_train_begin(self)``: called at beginning of first epoch of training
* ``on_train_end(self, metrics)``: called at end of training with metrics (see :ref:`section-metrics`) calculated during last epoch
* ``on_epoch_begin(self, epoch)``: called at the start of training epoch ``epoch``
* ``on_epoch_end(self, epoch, metrics)``: called at the start of training epoch ``epoch`` with metrics (see :ref:`section-metrics`) calculated during this epoch
* ``on_batch_begin(self, batch)``: called at the start of batch ``batch``
* ``on_batch_end(self, batch, metrics)``: called at the end of batch ``batch`` with the current metrics (see :ref:`section-metrics`) calculated during this batch
* ``on_timestep_begin(self, timestep)``: called at the start of timestep ``timestep``
* ``on_timestep_end(self, timestep)``: called at the end of timestep ``timestep``

.. note::
    These methods do not override methods in the base class but, for performance reasons, are detected by inspecting 
    callback objects.

To allow callback classes to access properties of the simulation, they can also provide a ``set_params`` method.
This method is called when callbacks are build into a :class:`~utils.callback_list.CallbackList` and is typically provided with the following keyword arguments:

* ``compiled_network``: the :class:`~compilers.CompiledNetwork`-derived object the network has been compiled into. This allows access to underlying GeNN model and thus it's neuron group and synapse group objects.
* ``num_batches``: number of batches per-epoch
* ``data``: dictionary used for storing recording data (see below)

All ``set_params`` methods should ignore unknown keyword arguments using a trailing ``kwargs**`` argument e.g.

..  code-block:: python

    def set_params(self, num_batches, **kwargs):
        pass

Saving data
^^^^^^^^^^^
When callback classes are used for recording, they should not directly store data themselves.
Instead, they should add data to the dictionary passed via the ``data`` keyword argument to ``set_params`` and provide a ``get_data`` method which returns the callback's key and the recorded data. For example, a callback which records to a list might do this:

..  code-block:: python

    from ml_genn.callbacks import Callback

    class MyCallback(Callback):
        def __init__(self, key):
            self.key = key

        def set_params(self, data, **kwargs):
            data[self.key] = []
            self._data = data[self.key]

        def on_batch_end(self, batch, metrics):
            # Append something to self._data
            pass

        def get_data(self):
            return self.key, self._data
