.. py:currentmodule:: ml_genn

.. _section-datasets:

Datasets
========
mlGeNN supports both spiking and non-spiking datasets.

Non-spiking
-----------
Non-spiking inputs are expected to be encoded as a numpy array with a shape whose 
first dimension is the number of examples in the dataset.

Static inputs
^^^^^^^^^^^^^
For static inputs which do not change throughout each example, the remaining dimensions of the array should 
match the shape of the input layer. For example, greyscale images from the MNIST training set might be encoded as 
a numpy array of shape :math:`(60000, 28, 28)` for a spiking convolutional network.
Static images are then encoded into a spatio-temporal spike train using an input neuron model.
For example, the :class:`~neurons.neuron.PoissonInput` turns a static input into a spike rate code with irregular spike train that have Poisson statistics.

Time-varying inputs
^^^^^^^^^^^^^^^^^^^
mlGeNN input neuron models also support time-varying inputs where the value to encode changes throughout the example.
These inputs require an additional dimension to specify frames. For example, the Google Speech Commands dataset,
with each utterance encoded as a a 40 channel MEL spectrograms with 80 time bins might be encoded as a numpy array
of shape :math:`(65000, 80, 40)` for a spiking recurrent network.

The input neuron model then needs to be correctly configured to match the number of time bins and 
the number of timesteps each one is presented for. For example, if we wanted to use each of our
MEL spectrogram time bins as an input current to a leaky integrate-and-fire neuron for 20 
timesteps, you would create an input layer like this:

.. code-block:: python

    input = Population(LeakyIntegrateFireInput(v_thresh=1, tau_mem=20,
                                               input_frames=80, input_frame_timesteps=20),
                       40)


Spiking
-------
Models using spiking datasets as input should use a :class:`.neurons.SpikeInput` input layer.
Inputs to these layers are expected to take the form of a list of :class:`ml_genn.utils.data.PreprocessedSpikes` objects.
These can be created from raw sequences of neuron ids and times using the :func:`.utils.data.preprocess_spikes` function.
For example, if we have an input population of four neurons and want them to fire, in sequence every 10ms you could could
create a :class:`ml_genn.utils.data.PreprocessedSpikes` object as follows:

.. code-block:: python

    from ml_genn.utils.data import preprocess_spikes
    
    spike_times = [0.0, 10.0, 20.0, 30.0]
    spike_ids = [0, 1, 2, 3]
    
    spikes = preprocess_spikes(spike_times, spike_ids, 4)

Alternatively, you can manually encode non-spiking data into :class:`ml_genn.utils.data.PreprocessedSpikes` using
either a linear or log latency code using :func:`.utils.data.linear_latency_encode_data` or 
:func:`.utils.data.log_latency_encode_data` respectively. For example to log latency encode the MNIST training 
dataset, you could do the following:

.. code-block:: python

    import mnist
    from ml_genn.utils.data import log_latency_encode_data

    labels = mnist.train_labels() 
    spikes = log_latency_encode_data(mnist.train_images(), 20.0, 51)

Finally, if you have a dataset from the popular `Tonic library <https://tonic.readthedocs.io/en/latest/>`_, mlGeNN provides
a helper function for converting datasets to :class:`ml_genn.utils.data.PreprocessedSpikes` format. For example, to convert
the training data from the Spiking Heidelberg Digits (SHD) dataset to mlGeNN format you would do the following:

.. code-block:: python

    from tonic.datasets import SHD
    from ml_genn.utils.data import preprocess_tonic_spikes
    
    dataset = SHD(save_to='../data', train=True)
    
    # Preprocess
    spikes = []
    labels = []
    for i in range(len(dataset)):
        events, label = dataset[i]
        spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                              dataset.sensor_size))
        labels.append(label)
