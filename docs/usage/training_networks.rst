.. py:currentmodule:: ml_genn

Training networks
=================
A key design goal of mlGeNN is that once a network topology has been defined using the API described in 
:ref:`section-building_networks`, it can be trained using *any* supported training algorithm.
:class:`.Network` and :class:`.SequentialNetwork` objects need to be *compiled* into
GeNN models for training using a training algorithm compiler class. Currently the 
following compilers for training networks are provided:

e-prop
------

.. autoclass:: ml_genn.compilers.EPropCompiler
    :noindex:

EventProp
---------
.. autoclass:: ml_genn.compilers.EventPropCompiler
    :noindex:
    
Once either compiler has been constructed, it can be used to compile a network with:

.. code-block:: python

    compiled_net = compiler.compile(network)

and this can be trained on a dataset, prepared as described in :ref:`section-datasets`, 
using a standard training loop with:

.. code-block:: python

    with compiled_net:  
        # Evaluate model on numpy dataset
        metrics, _  = compiled_net.train({input: spikes},
                                         {output: labels},
                                          num_epochs=50, shuffle=True)

Like in Keras, additional logic such as checkpointing and recording of 
state variables can be added to the standard training loop using callbacks 
as described in the :ref:`section-callbacks-recording` section.

Augmentation
------------
Like when using Keras, sometimes, merely adding callbacks to the standard
training loop is insufficient and you want to perform additional manual processing.
One common case of this is augmentation where you want to modify the data being trained
on each epoch. This can be implemented by manually looping over epochs and providing new 
data each time like this:

.. code-block:: python

    with compiled_net:
        for e in range(50):
            aug_spikes = augment(spikes)
            metrics, _  = compiled_net.train({input: aug_spikes},
                                             {output: labels},
                                             start_epoch=e, num_epochs=1,
                                             shuffle=True)

where ``augment`` is a function that returns an augmented version of a spike dataset (see :ref:`section-datasets`).

Default parameters
------------------
Sadly the mathematical derivation of the different training algorithms makes different
assumptions about the detailed implementation of various neuron models. For example, 
the e-prop learning rule assumes that neurons will have a 'relative reset'
where the membrane voltage has a fixed value subtracted from it after a spike whereas
EventProp assumes that the membrane voltage will be reset to a fixed value after a spike.
To avoid users having to remember these details, mlGeNN compilers provide a dictionary of
default parameters which can be passed to the constructors of :class:`.Network` 
and :class:`.SequentialNetwork`. For example here the e-prop defaults are applied to
a sequential network and hence the leaky integrate-and-fire layer within it:

..  code-block:: python

    from ml_genn import Layer, SequentialNetwork
    from ml_genn.neurons import LeakyIntegrateFire
    
    from ml_genn.compilers.eprop_compiler import default_params
    
    network = SequentialNetwork(default_params)
    with network:
        ...
        hidden = Layer(Dense(1.0), LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0, tau_refrac=5.0), 128)


