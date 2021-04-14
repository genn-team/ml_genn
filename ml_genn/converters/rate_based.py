import tensorflow as tf
import numpy as np

from enum import Enum
from tqdm import tqdm

from ml_genn.layers import IFNeurons
from ml_genn.layers import SpikeInputNeurons
from ml_genn.layers import PoissonInputNeurons
from ml_genn.layers import IFInputNeurons

class InputType(Enum):
    SPIKE = 'spike'
    SPIKE_SIGNED = 'spike_signed'
    POISSON = 'poisson'
    POISSON_SIGNED = 'poisson_signed'
    IF = 'if'

class NormMethod(Enum):
    DATA_NORM = 'data-norm'
    SPIKE_NORM = 'spike-norm'

class RateBased(object):
    def __init__(self, input_type=InputType.POISSON, norm_data=None,
                 norm_method=NormMethod.DATA_NORM, spike_norm_time=None):
        self.input_type = InputType(input_type)
        self.norm_data = norm_data
        self.norm_method = NormMethod(norm_method)
        self.spike_norm_time = spike_norm_time
        if self.norm_method == NormMethod.SPIKE_NORM and spike_norm_time is None:
            raise ValueError("When using spike-norm, "
                             "you must set spike norm time")

    def validate_tf_layer(self, tf_layer):
        if tf_layer.activation != tf.keras.activations.relu:
            raise NotImplementedError('{} activation not supported'.format(type(tf_layer.activation)))
        if tf_layer.use_bias == True:
            raise NotImplementedError('bias tensors not supported')

    def create_input_neurons(self, norm_output):
        if self.input_type == InputType.SPIKE:
            return SpikeInputNeurons()
        elif self.input_type == InputType.SPIKE_SIGNED:
            return SpikeInputNeurons(signed_spikes=True)
        elif self.input_type == InputType.POISSON:
            return PoissonInputNeurons()
        elif self.input_type == InputType.POISSON_SIGNED:
            return PoissonInputNeurons(signed_spikes=True)
        elif self.input_type == InputType.IF:
            return IFInputNeurons()

    def create_neurons(self, tf_layer, norm_output):
        return IFNeurons(threshold=1.0)
    
    def normalise_pre_compile(self, tf_model):
        # **TODO** data-norm could be implemented in this way
        pass

    def normalise_post_compile(self, tf_model, mlg_model):
        # If we should use data-normalisation
        if self.norm_data is not None and self.norm_method == NormMethod.DATA_NORM:
            # Get output functions for weighted layers.
            idx = [i for i, l in enumerate(tf_model.layers) 
                   if len(l.get_weights()) > 0]
            get_outputs = tf.keras.backend.function(
                tf_model.inputs, [tf_model.layers[i].output for i in idx])

            # Find the maximum activation in each layer, given input data.
            max_activation = np.array([np.max(out) for out in get_outputs(self.norm_data)], 
                                      dtype=np.float64)

            # Find the maximum weight in each layer.
            max_weights = np.array([np.max(w) for w in tf_model.get_weights()], 
                                   dtype=np.float64)

            # Compute scale factors and normalize weights.
            scale_factors = np.max([max_activation, max_weights], 0)
            applied_factors = np.empty(scale_factors.shape, dtype=np.float64)
            applied_factors[0] = scale_factors[0]
            applied_factors[1:] = scale_factors[1:] / scale_factors[:-1]

            # Update layer thresholds
            for layer, threshold in zip(mlg_model.layers[1:], applied_factors):
                print('layer <{}> threshold: {}'.format(layer.name, threshold))
                layer.neurons.set_threshold(threshold)
        # Otherwise, if we should use spike-norm
        elif self.norm_data is not None and self.norm_method == NormMethod.SPIKE_NORM:
            g_model = mlg_model.g_model
            n_samples = self.norm_data[0].shape[0]

            # Set layer thresholds high initially
            for layer in mlg_model.layers[1:]:
                layer.neurons.set_threshold(np.inf)

            # For each weighted layer
            for layer in mlg_model.layers[1:]:
                threshold = np.float64(0.0)

                # For each sample presentation
                progress = tqdm(total=n_samples)
                for batch_start in range(0, n_samples, g_model.batch_size):
                    batch_end = min(batch_start + g_model.batch_size, n_samples)
                    batch_n = batch_end - batch_start
                    batch_data = [x[batch_start:batch_end] 
                                  for x in self.norm_data]

                    # Set new input
                    mlg_model.reset()
                    mlg_model.set_input_batch(batch_data)

                    # Main simulation loop
                    while g_model.t < self.spike_norm_time:
                        # Step time
                        mlg_model.step_time()

                        # Get maximum activation
                        nrn = layer.neurons.nrn
                        nrn.pull_var_from_device('Vmem')
                        if nrn.vars['Vmem'].view.ndim == 1:
                            output_view = nrn.vars['Vmem'].view[np.newaxis]
                        else:
                            output_view = nrn.vars['Vmem'].view[:batch_n]
                        threshold = np.max([threshold, output_view.max()])
                        output_view[:] = np.float64(0.0)
                        nrn.push_var_to_device('Vmem')

                    progress.update(batch_n)

                progress.close()

                # Update this layer's threshold
                print('layer <{}> threshold: {}'.format(layer.name, threshold))
                layer.neurons.set_threshold(threshold)
