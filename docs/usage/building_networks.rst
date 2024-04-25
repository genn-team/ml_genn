.. py:currentmodule:: ml_genn

.. _section-building_networks:

Building networks
=================
One of the key aims of mlGeNN is to make it simple to define Spiking Neural Networks (SNNs)
with arbitrary topologies. Networks consist of homogeneous groups of neurons described by 
:class:`~population.Population` objects connected together with :class:`~connection.Connection` objects.
All populations and connections are owned by a :class:`Network` which acts as a
context manager so a network with two populations of neurons could simply be created like:

..  code-block:: python

    from ml_genn import Connection, Population, Network

    network = Network()
    with network:
        a = Population("poisson_input", 100)
        b = Population("integrate_fire", 100, readout="spike_count")
    
        Connection(a, b, Dense(1.0))

For simplicity, in this example, built-in neuron models with default parameters are 
specified using strings. However, if you wish to override some of the default model 
parameters, use a model that does not have default parameters or use a model not 
built into mlGeNN, you can also specify a neuron model using a :class:`~neurons.neuron.Neuron` 
class instance. For example, if we wished for the Poisson population to emit positive 
and negative spikes for positive and negative input values and for the integrate-and-fire 
neuron to have a higher firing threshold we could instantiate 
:class:`~neurons.poisson_input.PoissonInput` and :class:`~neurons.integrate_fire.IntegrateFire`
objects ourselves like:

..  code-block:: python

    from ml_genn import Connection, Population, Network
    from ml_genn.neurons import PoissonInput, IntegrateFire

    network = Network()
    with network:
        a = Population(PoissonInput(signed_spikes=True), 100)
        b = Population(IntegrateFire(v_thresh=2.0), 100, readout="spike_count")
    
        Connection(a, b, Dense(1.0))

By default, :class:`~connection.Connection` objects use a 'delta' synapse model where the 
accumulated weight of incoming spikes is directly injected into neurons. However, if
you wish to use a somewhat more realistic model where inputs are *shaped* to mimic the 
dynamics of real ion channels, this can be replaced. 
This same general principle is also used for specifying and configuring many other aspects of mlGeNN networks,
including :mod:`ml_genn.losses`, :mod:`ml_genn.metrics` and :mod:`ml_genn.readouts`.

Sequential networks
-------------------
While the flexibility to create networks with any topology is very useful,
feed-forward networks are very common so mlGeNN provides a shorthand syntax for
specifying them more tersely:

..  code-block:: python

    from ml_genn import InputLayer, Layer, SequentialNetwork

    network = SequentialNetwork()
    with network:
        a = InputLayer("poisson_input", 100)
        b = Layer(Dense(1.0), "integrate_fire", 100, readout="spike_count")

