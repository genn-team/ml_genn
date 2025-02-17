mlGeNN documentation
====================
mlGeNN is a new library for machine learning with Spiking Neural Networks (SNNs),
built on the efficient foundation provided by `our GeNN simulator <https://github.com/genn-team/genn>`_.
mlGeNN expose the constructs required to build SNNs using an
API, inspired by modern ML libraries like `Keras <https://keras.io/>`_, which aims 
to reduce cognitive load  by automatically calculating layer sizes, default 
hyperparameter values etc to enable rapid prototyping of SNN models.

Why another SNN library
-----------------------
While there are already a plethora of SNN simulators, most are designed for
Computational Neuroscience applications and, as such, not only provide unfamiliar
abstractions for ML researchers but also don't support standard ML workflows
such as data-parallel batch training. 
Because of this, researchers have chosen to stick with familiar frameworks such as PyTorch 
and built libraries to adapt them for SNNs such as `BindsNET <https://github.com/BindsNET/bindsnet>`_, 
`NORSE <https://github.com/norse/norse>`_, `SNNTorch <https://snntorch.readthedocs.io/en/latest/>`_ 
and `Spiking Jelly <https://github.com/fangwei123456/spikingjelly>`_.

However, these libraries are all constrained by the underlying nature of ML frameworks 
where the activity of populations of neurons is typically represented as a vector of 
activities and, for an SNN, this vector is populated with ones for spiking and zeros for non-spiking neurons. 
This representation allows one to apply the existing infrastructure of the underlying 
ML framework to SNNs but, as spiking neurons often spike at comparatively low rates, 
propagating the activity of inactive neurons through the network leads to 
unnecessary computation.

mlGeNN provides user friendly implementations of novel SNN training algorithms
such as e-prop [Bellec2020]_ and EventProp [Wunderlich2021]_ to enable spike-based ML 
on top of GeNN’s GPU-optimised sparse data structures and algorithms.

.. toctree::
    :maxdepth: 3
    :titlesonly:

    usage/building_networks
    usage/datasets
    usage/training_networks
    usage/inference
    usage/callbacks_recording
    usage/metrics
    usage/converting_tf
    usage/bibliography
    
    tutorials/index

    mlGeNN reference <source/ml_genn>
    mlGeNN TF reference <source/ml_genn_tf>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
