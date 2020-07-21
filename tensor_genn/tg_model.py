"""TensorGeNN model definition

This module provides TGModel class to convert TensorFlow models into GeNN
models, and provides helper functions for operating the resulting GeNN model.

A ``TGModel`` object can use a pre-trained TensorFlow model to function.
Such a model can be provided by calling the ``convert_tf_model`` method
with the TensorFlow model and optional parameters.

Example:
    The following is a minimal example which demonstrates the process of
    converting a TensorFlow model into a GeNN model and evaluating it:

        from tensor_genn import TGModel

        tensorgenn_model = TGmodel()
        tensorgenn_model.convert_tf_model(tensorflow_model)
        tensorgenn_model.compile()
        tensorgenn_model.evaluate(test_data, test_labels)
"""

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from pygenn.genn_model import GeNNModel

from tensor_genn.layers import InputType, Input, SpikeInput, PoissonInput, IFInput
from tensor_genn.layers import Dense, IFDense
from tensor_genn.layers import Conv2D, IFConv2D


class TGModel(object):
    """TensorGeNN model class

    This class converts fully trained TensorFlow models into GeNN models,
    and provides an interface for manipulating converted models.
    """

    def __init__(self, name='tg_model'):
        """Initialise a TensorGeNN model"""

        self.name = name
        self.layers = []
        self.inputs = []
        self.outputs = []

        self.g_model = None
        self.batch_size = None
        self.share_weights = None


    def convert_tf_model(self, tf_model, input_type='poisson'):
        """Convert from a TensorFlow model

        Args:
        tf_model  --  TensorFlow model to be converted

        Keyword args:
        input_type     --  type of input neurons (default: 'poisson')
        """

        supported_tf_layers = (
            tf.keras.layers.Dense,
            tf.keras.layers.Conv2D,
            tf.keras.layers.AveragePooling2D,
            tf.keras.layers.Flatten,
            tf.keras.layers.Dropout,
        )

        # Check model compatibility
        if not isinstance(tf_model, tf.keras.Sequential):
            raise NotImplementedError('{} models not supported'.format(type(tf_model)))
        for tf_layer in tf_model.layers[:-1]:
            if not isinstance(tf_layer, supported_tf_layers):
                raise NotImplementedError('{} layers not supported'.format(type(tf_layer)))
            elif isinstance(tf_layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                if tf_layer.activation != tf.keras.activations.relu:
                    raise NotImplementedError('{} activation not supported'.format(type(tf_layer.activation)))
                if tf_layer.use_bias == True:
                    raise NotImplementedError('bias tensors not supported')


        self.name = tf_model.name
        self.inputs = []
        self.outputs = []
        self.layers = []


        # Add input layer
        input_type = InputType(input_type)
        if input_type == InputType.SPIKE:
            layer = SpikeInput('input', tf_model.input_shape[1:])
        elif input_type == InputType.POISSON:
            layer = PoissonInput('input', tf_model.input_shape[1:])
        elif input_type == InputType.IF:
            layer = IFInput('input', tf_model.input_shape[1:])

        self.inputs.append(layer)
        self.layers.append(layer)
        previous_layer = layer


        # For each TensorFlow model layer:
        for tf_layer in tf_model.layers:

            # === Flatten Layers ===
            if isinstance(tf_layer, tf.keras.layers.Flatten):
                print('ignoring Flatten layer <{}>'.format(tf_layer.name))
                continue

            # === Dropout Layers ===
            elif isinstance(tf_layer, tf.keras.layers.Dropout):
                print('ignoring Dropout layer <{}>'.format(tf_layer.name))
                continue

            # === Dense Layers ===
            elif isinstance(tf_layer, tf.keras.layers.Dense):
                print('converting Dense layer <{}>'.format(tf_layer.name))

                layer = IFDense(tf_layer.name, tf_layer.units)
                layer.connect([previous_layer])
                layer.set_weights(tf_layer.get_weights())

                self.layers.append(layer)
                previous_layer = layer


            # === Conv2D Layers ===
            elif isinstance(tf_layer, tf.keras.layers.Conv2D):
                print('converting Conv2D layer <{}>'.format(tf_layer.name))

                layer = IFConv2D(tf_layer.name, tf_layer.filters, tf_layer.kernel_size, tf_layer.strides, tf_layer.padding)
                layer.connect([previous_layer])
                layer.set_weights(tf_layer.get_weights())

                self.layers.append(layer)
                previous_layer = layer


            # # === AveragePooling2D Layers ===
            # elif isinstance(tf_layer, tf.keras.layers.AveragePooling2D):
            #     print('defer weights for layer <{}>'.format(tf_layer.name))
            #     g_vals = np.zeros((np.prod(tf_layer.input_shape[1:]), np.prod(tf_layer.output_shape[1:])))
            #     g_conn = np.zeros(g_vals.shape, dtype=np.bool)
            #     defer_weights = True

            #     ph, pw = tf_layer.pool_size
            #     sh, sw = tf_layer.strides
            #     ih, iw, ic = tf_layer.input_shape[1:]
            #     oh, ow, oc = tf_layer.output_shape[1:]

            #     # Stride start index iterators.
            #     if tf_layer.padding == 'valid':
            #         stride_rows = range(0, ih - ph + 1, sh)
            #         stride_cols = range(0, iw - pw + 1, sw)
            #     elif tf_layer.padding == 'same':
            #         stride_rows = range(0 - ph // 2, ih - ph // 2, sh)
            #         stride_cols = range(0 - pw // 2, iw - pw // 2, sw)

            #     # For each pool (one-to-one input -> output channels):
            #     for channel in range(ic):
            #         g_pool_vals = g_vals[channel::ic, channel::ic]
            #         g_pool_conn = g_conn[channel::ic, channel::ic]

            #         # For each output neuron:
            #         for out_row, stride_row in enumerate(stride_rows):
            #             for out_col, stride_col in enumerate(stride_cols):
            #                 g_out_vals = g_pool_vals[:, out_row * ow + out_col]
            #                 g_out_vals.shape = (ih, iw)
            #                 g_out_conn = g_pool_conn[:, out_row * ow + out_col]
            #                 g_out_conn.shape = (ih, iw)

            #                 # Get a pool stride view in g_out_vals.
            #                 g_pool_T = max(stride_row, 0)
            #                 g_pool_B = min(stride_row + ph, ih)
            #                 g_pool_L = max(stride_col, 0)
            #                 g_pool_R = min(stride_col + pw, iw)
            #                 g_stride_vals = g_out_vals[g_pool_T:g_pool_B, g_pool_L:g_pool_R]
            #                 g_stride_conn = g_out_conn[g_pool_T:g_pool_B, g_pool_L:g_pool_R]

            #                 # Set weights for this stride.
            #                 g_stride_vals[:] = 1.0 / g_stride_vals.size
            #                 g_stride_conn[:] = True

            # # === Combine Deferred Weights ===
            # if deferred_vals is not None:
            #     print('combining deferred weights with GeNN weights for layer <{}>'.format(tf_layer.name))
            #     new_vals = np.zeros((deferred_vals.shape[0], g_vals.shape[1]))
            #     new_conn = np.zeros(new_vals.shape, dtype=np.bool)

            #     # For each input channel:
            #     for in_channel in range(ic):
            #         # Note: it is assumed that the deferred weight matrix maps input
            #         # channel i one-to-one to output channel i, for all i in [0, ic).
            #         new_in_channel_vals = new_vals[in_channel::ic, :]
            #         new_in_channel_conn = new_conn[in_channel::ic, :]
            #         deferred_in_channel_vals = deferred_vals[in_channel::ic, in_channel::ic]
            #         deferred_in_channel_conn = deferred_conn[in_channel::ic, in_channel::ic]
            #         g_in_channel_vals = g_vals[in_channel::ic, :]
            #         g_in_channel_conn = g_conn[in_channel::ic, :]

            #         # Set weights to dot product of deferred and new weights.
            #         new_in_channel_vals[:] = np.dot(deferred_in_channel_vals, g_in_channel_vals)
            #         new_in_channel_conn[:] = np.dot(deferred_in_channel_conn, g_in_channel_conn)

            #     # Update weights.
            #     g_vals = new_vals
            #     g_conn = new_conn

            # # === Append Weights ===
            # if defer_weights:
            #     # Defer weights to next layer.
            #     deferred_vals = g_vals
            #     deferred_conn = g_conn
            # else:
            #     # Append weights for this layer.
            #     self.layer_names.append(tf_layer.name)
            #     self.weight_vals.append(g_vals)
            #     self.weight_conn.append(g_conn)
            #     self.thresholds.append(np.float64(1.0))
            #     deferred_vals = None
            #     deferred_conn = None

        self.outputs.append(previous_layer)


    def set_network(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = []

        # Construct topologically sorted list of layers
        new_layers = set(inputs)
        seen_connections = set()
        while new_layers:
            layer = new_layers.pop()
            self.layers.append(layer)

            # Explore downstream layers whose upstream connections have all been seen
            for downstream_connection in layer.downstream_connections:
                seen_connections.add(downstream_connection)
                if seen_connections.issuperset(downstream_connection.target.upstream_connections):
                    new_layers.add(downstream_connection.target)

        # Check that output layers are reachable from input layers
        if not all(output in self.layers for output in self.outputs):
            raise ValueError('output layers unreachable from input layers')


    def compile(self, dt=1.0, rng_seed=0, batch_size=1, share_weights=False):
        """Compile this TensorGeNN model into a GeNN model

        Keyword args:
        dt             --  model integration time step (default: 1.0)
        rng_seed       --  GeNN RNG seed (default: 0, meaning choose a random seed)
        batch_size     --  number of models to run concurrently (default: 1)
        share_weights  --  share weights within model batch (default: False)
        """

        # Define GeNN model
        self.g_model = GeNNModel('float', self.name)
        self.g_model.timing_enabled = True
        self.g_model.dT = dt
        self.g_model._model.set_seed(rng_seed)
        self.batch_size = batch_size
        self.share_weights = share_weights

        # Prepare each layer
        for layer in self.layers:
            layer.compile(self)

        # Build and load model
        self.g_model.build()
        self.g_model.load()


    def set_input_batch(self, data_batch):
        """Set model input with a new batch of samples

        Args:
        data_batch  --  list of data batches for each input layer
        """

        # Input sanity check
        if len(data_batch) != len(self.inputs):
            raise ValueError('data batch list length and input layer list length mismatch')

        for i in range(len(self.inputs)):
            self.inputs[i].set_input_batch(data_batch[i])


    def step_time(self, iterations=1):
        """Iterate the GeNN model a given number of steps

        Keyword args:
        iterations  --  number of iterations (default: 1)
        """

        for i in range(iterations):
            self.g_model.step_time()


    def reset(self):
        """Reset the GeNN model"""

        self.g_model.timestep = 0
        self.g_model.t = 0.0


    def evaluate(self, data, labels, time, save_samples=[]):
        """Evaluate the accuracy of a GeNN model

        Args:
        data          --  list of data for each input layer
        labels        --  list of labels for each output layer
        time          --  sample present time (msec)

        Keyword args:
        save_samples  --  list of sample indices to save spikes for (default: [])

        Returns:
        accuracy      --  percentage of correctly classified results
        spike_i       --  list of spike indices for each sample index in save_samples
        spike_t       --  list of spike times for each sample index in save_samples
        """

        # Input sanity check
        n_samples = data[0].shape[0]
        save_samples = list(set(save_samples))
        if len(data) != len(self.inputs):
            raise ValueError('data list length and input layer list length mismatch')
        if len(labels) != len(self.outputs):
            raise ValueError('label list length and output layer list length mismatch')
        if not all(x.shape[0] == n_samples for x in data + labels):
            raise ValueError('sample count mismatch in data and labels arrays')
        if any(i < 0 or i >= n_samples for i in save_samples):
            raise ValueError('one or more invalid save_samples value')




        print('remaining CUDA memory (GB):  ' + str(self.g_model.free_device_mem_bytes / (1024 * 1024 * 1024)))

        import time as t
        t0_kernel = sum(self.get_kernel_times().values())
        t0_clock = t.time()




        n_correct = [0] * len(self.outputs)
        accuracy = [0] * len(self.outputs)
        all_spikes = [[[]] * len(self.layers)] * len(save_samples)

        # Process sample batches
        progress = tqdm(total=n_samples)
        for batch_start in range(0, n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_samples)
            batch_data = [x[batch_start:batch_end] for x in data]
            batch_labels = [y[batch_start:batch_end] for y in labels]
            save_samples_in_batch = [i for i in save_samples if batch_start <= i < batch_end]

            # Set new input
            self.reset()
            self.set_input_batch(batch_data)

            # Main simulation loop
            while self.g_model.t < time:

                # Step time
                self.step_time()

                # Save spikes
                for i in save_samples_in_batch:
                    k = save_samples.index(i)
                    batch_i = i - batch_start
                    for l, layer in enumerate(self.layers):
                        nrn = layer.nrn[batch_i]
                        nrn.pull_current_spikes_from_device()
                        all_spikes[k][l].append(np.copy(nrn.current_spikes))

            # Compute accuracy
            for output_i in range(len(self.outputs)):
                for batch_i in range(batch_end - batch_start):
                    nrn = self.outputs[output_i].nrn[batch_i]
                    nrn.pull_var_from_device('nSpk')
                    n_correct[output_i] += nrn.vars['nSpk'].view.argmax() == batch_labels[batch_i]

                accuracy[output_i] = (n_correct[output_i] / batch_end) * 100

            progress.set_postfix_str('accuracy: {:2.2f}'.format(np.mean(accuracy)))
            progress.update(batch_end - batch_start)

        progress.close()





        t_clock = t.time() - t0_clock
        t_kernel = sum(self.get_kernel_times().values()) - t0_kernel

        print('batch_size  clock_time  kernel_time  kernel_ratio')
        print('{}  {}  {}  {}'.format(self.batch_size, t_clock, t_kernel ,t_kernel / t_clock))



        # Create spike index and time lists
        spike_i = [[[]] * len(self.layers)] * len(save_samples)
        spike_t = [[[]] * len(self.layers)] * len(save_samples)
        for i in range(len(save_samples)):
            for j in range(len(self.layers)):
                spikes = all_spikes[i][j]
                spike_i[i][j] = np.concatenate(spikes)
                spike_t[i][j] = np.concatenate([np.ones_like(s) * i * self.g_model.dt for i, s in enumerate(spikes)])

        return accuracy, spike_i, spike_t


    def get_kernel_times(self):
        """Get total kernel run times"""

        return {
            'init_time': self.g_model.init_time,
            'init_sparse_time': self.g_model.init_sparse_time,
            'neuron_update_time': self.g_model.neuron_update_time,
            'presynaptic_update_time': self.g_model.presynaptic_update_time,
            'postsynaptic_update_time': self.g_model.postsynaptic_update_time,
            'synapse_dynamics_time': self.g_model.synapse_dynamics_time,
        }
