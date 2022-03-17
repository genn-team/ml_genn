[![Build Status](https://gen-ci.inf.sussex.ac.uk/buildStatus/icon?job=GeNN/ml_genn/master)](https://gen-ci.inf.sussex.ac.uk/job/GeNN/job/ml_genn/job/master/)
# ML GeNN
A library for deep learning with Spiking Neural Networks (SNN)powered by [GeNN](http://genn-team.github.io/genn/), a GPU enhanced Neuronal Network simulation environment.

## Installation
 1. Follow the instructions in https://github.com/genn-team/genn/blob/master/pygenn/README.md to install PyGeNN.
 2. Clone this project
 3. Install with setuptools using ``python setup.py develop`` command

## Usage
The following example illustrates how to convert an ANN model, defined in Keras, to an SNN using the few-spike ([Stöckl & Maass, 2021](http://dx.doi.org/10.1038/s42256-021-00311-4)) conversion method:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
tf_model = Sequential([
   Conv2D(16, 3, padding='same', activation='relu',
          use_bias=False, input_shape=(32, 32, 3)),
   AveragePooling2D(2),
   Flatten(),
   Dense(10, activation='relu', use_bias=False)])

# compile and train tf_model with TensorFlow
...

from ml_genn import Model
from ml_genn.converters import FewSpike
converter = FewSpike([x_subset], signed_input=True)
tg_model = Model.convert_tf_model(tf_model, converter=converter,
                                  connectivity_type='toeplitz', 
                                  dt=1.0, batch_size=50, 
                                  rng_seed=0)
tg_model.evaluate([x], [y], converter.K)
```
For further examples, please see the examples folder.
