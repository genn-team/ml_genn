import tensorflow as tf
import numpy as np
from tqdm import tqdm

from ml_genn.layers import InputType
from ml_genn.layers import IFNeurons
from ml_genn.layers import SpikeInputNeurons
from ml_genn.layers import PoissonInputNeurons
from ml_genn.layers import IFInputNeurons

class SpikeNorm(object):
    def __init__(self, norm_data, norm_time, signed_input=False, 
                 input_type=InputType.POISSON):
        self.norm_data = norm_data
        self.norm_time = norm_time
        self.signed_input = signed_input
        self.input_type = InputType(input_type)

    def validate_tf_layer(self, tf_layer, config):
        if isinstance(tf_layer, (
                tf.keras.layers.Dense,
                tf.keras.layers.Conv2D)):

            if tf_layer.use_bias:
                # no bias tensors allowed
                raise NotImplementedError('Spike-Norm converter: bias tensors not supported')

            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError(
                        'Spike-Norm converter: output layer must have ReLU or softmax activation')

            elif config.has_activation:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError(
                        'Spike-Norm converter: hidden layers must have ReLU activation')

        elif isinstance(tf_layer, tf.keras.layers.Activation):
            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError(
                        'Spike-Norm converter: output layer must have ReLU or softmax activation')

            else:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError(
                        'Spike-Norm converter: hidden layers must have ReLU activation')

        elif isinstance(tf_layer, tf.keras.layers.ReLU):
            # ReLU activation allowed everywhere
            pass

        elif isinstance(tf_layer, tf.keras.layers.Softmax):
            # softmax activation only allowed for output layers
            if not config.is_output:
                raise NotImplementedError(
                    'Spike-Norm converter: only output layers may use softmax')

        elif isinstance(tf_layer, (
                tf.keras.layers.AveragePooling2D,
                tf.keras.layers.GlobalAveragePooling2D)):
            # average pooling allowed
            pass

        else:
            # no other layers allowed
            raise NotImplementedError(
                'Spike-Norm converter: {} layers are not supported'.format(
                    tf_layer.__class__.__name__))

    def create_input_neurons(self, pre_convert_output):
        if self.input_type == InputType.SPIKE:
            return SpikeInputNeurons(signed_spikes=self.signed_input)
        elif self.input_type == InputType.POISSON:
            return PoissonInputNeurons(signed_spikes=self.signed_input)
        elif self.input_type == InputType.IF:
            return IFInputNeurons()

    def create_neurons(self, tf_layer, pre_convert_output):
        return IFNeurons(threshold=1.0)

    def pre_convert(self, tf_model):
        pass
    
    def pre_compile(self, mlg_model):
        pass

    def post_compile(self, mlg_model):
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
                while g_model.t < self.norm_time:
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
