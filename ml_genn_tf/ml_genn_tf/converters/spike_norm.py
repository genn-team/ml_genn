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
        
        elif isinstance(tf_layer, tf.keras.layers.GlobalAveragePooling2D):
            # global average pooling allowed
            pass
        elif isinstance(tf_layer, tf.keras.layers.AveragePooling2D):
            if tf_layer.padding != 'valid':
                raise NotImplementedError(
                    'Spike-Norm converter: only valid padding is supported for pooling layers')

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

        # Don't allow models with multiple input layers
        if len(mlg_model.inputs) != 1:
            raise NotImplementedError(
                'Spike-Norm converter: models with multiple input layers not supported')

        final_thresholds = {}

        # Set layer thresholds high initially
        for layer in mlg_model.layers[1:]:
            if layer in mlg_model.inputs:
                continue

            layer.neurons.set_threshold(np.inf)
            final_thresholds[layer] = np.float64(1.0)

        # For each layer (these *should* be topologically sorted)
        for layer in mlg_model.layers:
            if layer in mlg_model.inputs:
                continue

            # Break at branch (many outbound)
            if len(layer.downstream_synapses) > 1:
                break

            norm_data_iter = iter(self.norm_data[0])

            threshold = np.float64(0.0)

            # For each sample presentation
            progress = tqdm(leave=False)
            for batch_data, _ in norm_data_iter:
                progress.set_description(f'layer <{layer.name}>')

                batch_data = np.array(batch_data)
                batch_size = batch_data.shape[0]

                # Set new input
                mlg_model.reset()
                mlg_model.set_input_batch([batch_data])

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
                        output_view = nrn.vars['Vmem'].view[:batch_size]
                    threshold = np.max([threshold, output_view.max()])
                    output_view[:] = np.float64(0.0)
                    nrn.push_var_to_device('Vmem')

                progress.update(batch_size)

            progress.close()

            # Update this layer's threshold
            layer.neurons.set_threshold(threshold)
            final_thresholds[layer] = threshold

        # For each layer (these *should* be topologically sorted)
        for layer in mlg_model.layers:
            if layer in mlg_model.inputs:
                continue

            # Update this layer's threshold
            layer.neurons.set_threshold(final_thresholds[layer])
            print('layer <{}> threshold: {}'.format(layer.name, layer.neurons.threshold))