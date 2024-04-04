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

To provid
While previous libraries for spike-based ML such as BindsNET22 have allowed SNNs to 
be defined in a familiar environment for ML researchers, they have been implemented 
on top of libraries like PyTorch. When using these libraries, the activity of a 
population of neurons is typically represented as a vector of rates and, for an SNN, 
this vector is populated with ones for spiking and zeros for non-spiking neurons. 
This representation allows one to apply the existing infrastructure of the underlying 
ML library to SNNs but, as real neurons often spike at comparatively low rates, 
propagating the activity of inactive neurons through the network leads to 
unnecessary computation.

mlGeNN aims to provi, spike-based ML which harnesses GeNN’s GPU-optimised sparse data
structures and algorithms.

mlGeNN 
researchers have chosen to stick with familiar tools such as PyTorch which are 
highly-optimised for rate-based models, but do not take advantage of the 
spatio-temporal sparsity of SNNs which have the potential to enable massive 
computational savings over rate-based networks17. 


.. toctree::
    :maxdepth: 3
    :titlesonly:

    usage/building_networks
    usage/callbacks_recording
    usage/metrics
    usage/converting_tf

    mlGeNN reference <source/ml_genn>
    mlGeNN TF reference <source/ml_genn_tf>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
