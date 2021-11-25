from tqdm import tqdm
import time as t
import numpy as np
from typing import Iterable, Optional, Union, Tuple, Sequence

import tensorflow as tf
from tensorflow.keras import models, layers, datasets, regularizers, optimizers

from ml_genn import Model
from ml_genn.converters import DataNorm
from ml_genn.utils import parse_arguments

import torch

import bindsnet.network as bnn
from bindsnet.datasets import DataLoader
from bindsnet.encoding import NullEncoder, PoissonEncoder

def initializer(shape, dtype=None):
    stddev = np.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)

class AvgPool2dConnection(bnn.topology.AbstractConnection):
    # language=rst
    """
    Specifies max-pooling synapses between one or two populations of neurons by keeping online estimates of maximally
    firing neurons.
    """

    def __init__(
        self,
        source: bnn.nodes.Nodes,
        target: bnn.nodes.Nodes,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Instantiates a ``AvgPool2dConnection`` object.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param padding: Horizontal and vertical padding for convolution.
        Keyword arguments:
        :param decay: Decay rate of online estimates of average firing activity.
        """
        super().__init__(source, target, None, None, 0.0, **kwargs)

        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)
        self.padding = torch.nn.modules.utils._pair(padding)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute max-pool pre-activations given spikes using online firing rate estimates.
        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without decaying spike activation).
        """

        return torch.nn.functional.avg_pool2d(
            s.float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # return s.take(indices).float()

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        No weights -> no normalization.
        """
        pass

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

class PassThroughNodes(bnn.nodes.Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_ with using reset by
    subtraction.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Sequence[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons.
        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )
        self.register_buffer("v", torch.zeros(self.shape))

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.
        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        self.s = x

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        self.s.zero_()

with tf.device('/CPU:0'):
    args = parse_arguments('VGG16 classifier model')
    print('arguments: ' + str(vars(args)))

    classify_time = 500
    dt = args.dt
    batch_size = args.batch_size

    # Retrieve and normalise CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_test = x_test[:args.n_test_samples] / 255.0
    #x_test -= np.average(x_test)
    y_test = y_test[:args.n_test_samples, 0]
    x_norm = x_train[np.random.choice(x_train.shape[0], args.n_norm_samples, replace=False)]

    # Check input size
    if x_train.shape[1] < 32 or x_train.shape[2] < 32:
        raise ValueError('input must be at least 32x32')

    # TensorFlow model
    print('TENSORFLOW MODEL')

    # Create L2 regularizer
    regularizer = regularizers.l2(0.0001)

    # Create, train and evaluate TensorFlow model
    tf_model = models.Sequential([
        # 32x32
        layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, input_shape=x_train.shape[1:], 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.3),
        layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        # 16x16
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        # 8x8
        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        # 4x4
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        # 2x2
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        layers.Flatten(),
        layers.Dense(4096, activation="relu", use_bias=False, kernel_regularizer=regularizer),
        layers.Dropout(0.5),
        layers.Dense(4096, activation="relu", use_bias=False, kernel_regularizer=regularizer),
        layers.Dropout(0.5),
        layers.Dense(y_train.max() + 1, activation="softmax", use_bias=False, kernel_regularizer=regularizer),
    ], name='vgg16_bindsnet')

    optimizer = optimizers.SGD(lr=0.05, momentum=0.9)

    tf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    tf_model.load_weights('vgg16_tf_weights.h5')

    # Create a suitable converter to convert TF model to ML GeNN
    # **THINK** BindsNet will not be signed
    converter = DataNorm(x_norm, signed_input=False)

    # Convert and compile ML GeNN model
    mlg_model = Model.convert_tf_model(
        tf_model, converter=converter, connectivity_type=args.connectivity_type,
        dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed, 
        kernel_profiling=args.kernel_profiling)

    # BindsNET model
    print('BINDSNET MODEL')

    bn_model = bnn.Network()

    # convert input to pytorch channel first from tf channel last
    x_test_chan_first = np.moveaxis(x_test, 3, 1)

    # convert conv weights to pytorch format [out_ch, in_ch, h, w] from tf format [h, w, in_ch, out_ch]
    syn_conv_w = [np.moveaxis(tf_model.get_weights()[i], (3, 2), (0, 1))
                  for i in range(13)]

    # convert dense weights to handle pytorch input format (channels first)
    syn_dense_1_w = np.empty(tf_model.get_weights()[13].shape, dtype=tf_model.get_weights()[13].dtype)
    for i in range(512):
        syn_dense_1_w[(i * 4 * 4):((i + 1) * 4 * 4)] = tf_model.get_weights()[13][i::512]

    input_layer = bnn.nodes.Input(n=(3 * 32 * 32), shape=(3, 32, 32))
    bn_model.add_layer(layer=input_layer, name='input')

    # Block 1
    conv_layer_1 = bnn.nodes.IFNodes(n=(64 * 32 * 32), shape=(64, 32, 32), thresh=641.8155517578125, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_1, name='conv_1')

    conv_layer_2 = bnn.nodes.IFNodes(n=(64 * 32 * 32), shape=(64, 32, 32), thresh=3.6156136989593506, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_2, name='conv_2')

    pass_layer_1 = PassThroughNodes(n=(64 * 16 * 16), shape=(64, 16, 16))
    bn_model.add_layer(layer=pass_layer_1, name='pass_1')

    # Block 2
    conv_layer_3 = bnn.nodes.IFNodes(n=(128 * 16 * 16), shape=(128, 16, 16), thresh=0.43528640270233154, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_3, name='conv_3')

    conv_layer_4 = bnn.nodes.IFNodes(n=(128 * 16 * 16), shape=(128, 16, 16), thresh=1.3611167669296265, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_4, name='conv_4')

    pass_layer_2 = PassThroughNodes(n=(128 * 8 * 8), shape=(128, 8, 8))
    bn_model.add_layer(layer=pass_layer_2, name='pass_2')

    # Block 3
    conv_layer_5 = bnn.nodes.IFNodes(n=(256 * 8 * 8), shape=(256, 8, 8), thresh=0.34528371691703796, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_5, name='conv_5')

    conv_layer_6 = bnn.nodes.IFNodes(n=(256 * 8 * 8), shape=(256, 8, 8), thresh=0.9875544905662537, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_6, name='conv_6')

    conv_layer_7 = bnn.nodes.IFNodes(n=(256 * 8 * 8), shape=(256, 8, 8), thresh=1.470356822013855, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_7, name='conv_7')

    pass_layer_3 = PassThroughNodes(n=(256 * 4 * 4), shape=(256, 4, 4))
    bn_model.add_layer(layer=pass_layer_3, name='pass_3')

    # Block 4
    conv_layer_8 = bnn.nodes.IFNodes(n=(512 * 4 * 4), shape=(512, 4, 4), thresh=0.3519073724746704, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_8, name='conv_8')

    conv_layer_9 = bnn.nodes.IFNodes(n=(512 * 4 * 4), shape=(512, 4, 4), thresh=1.2192555665969849, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_9, name='conv_9')

    conv_layer_10 = bnn.nodes.IFNodes(n=(512 * 4 * 4), shape=(512, 4, 4), thresh=1.4588868618011475, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_10, name='conv_10')

    pass_layer_4 = PassThroughNodes(n=(512 * 2 * 2), shape=(512, 2, 2))
    bn_model.add_layer(layer=pass_layer_4, name='pass_4')

    # Block 5
    conv_layer_11 = bnn.nodes.IFNodes(n=(512 * 2 * 2), shape=(512, 2, 2), thresh=0.5751084089279175, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_11, name='conv_11')

    conv_layer_12 = bnn.nodes.IFNodes(n=(512 * 2 * 2), shape=(512, 2, 2), thresh=1.98782217502594, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_12, name='conv_12')

    conv_layer_13 = bnn.nodes.IFNodes(n=(512 * 2 * 2), shape=(512, 2, 2), thresh=3.515345573425293, reset=0.0, refac=0)
    bn_model.add_layer(layer=conv_layer_13, name='conv_13')

    pass_layer_5 = PassThroughNodes(n=(512 * 1 * 1), shape=(512, 1, 1))
    bn_model.add_layer(layer=pass_layer_5, name='pass_5')

    # Classifier
    dense_layer_1 = bnn.nodes.IFNodes(n=4096, thresh=0.39055076241493225, reset=0.0, refac=0)
    bn_model.add_layer(layer=dense_layer_1, name='dense_1')

    dense_layer_2 = bnn.nodes.IFNodes(n=4096, thresh=1.4990026950836182, reset=0.0, refac=0)
    bn_model.add_layer(layer=dense_layer_2, name='dense_2')

    dense_layer_3 = bnn.nodes.IFNodes(n=10, thresh=0.0009850480128079653, reset=0.0, refac=0)
    bn_model.add_layer(layer=dense_layer_3, name='dense_3')

    # Make convolutional connections
    for src, src_name, trg, trg_name, w in zip([input_layer, conv_layer_1, pass_layer_1, conv_layer_3, pass_layer_2, conv_layer_5, conv_layer_6, pass_layer_3, conv_layer_8, conv_layer_9, pass_layer_4, conv_layer_11, conv_layer_12],
                                               ["input", "conv_1", "pass_1", "conv_3", "pass_2", "conv_5", "conv_6", "pass_3", "conv_8", "conv_9", "pass_4", "conv_11", "conv_12"],
                                               [conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_4, conv_layer_5, conv_layer_6, conv_layer_7, conv_layer_8, conv_layer_9, conv_layer_10, conv_layer_11, conv_layer_12, conv_layer_13],
                                               ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5", "conv_6", "conv_7", "conv_8", "conv_9", "conv_10", "conv_11", "conv_12", "conv_13"],
                                               syn_conv_w):
        syn_conv = bnn.topology.Conv2dConnection(source=src, target=trg, 
                                                 kernel_size=3, padding=1,
                                                 w=torch.tensor(w))
        bn_model.add_connection(connection=syn_conv, source=src_name, target=trg_name)
        assert (syn_conv.w.numpy() == w).all()
        assert (syn_conv.b == 0.0).all()

    # Make Avg pool connections
    for src, src_name, trg, trg_name in zip([conv_layer_2, conv_layer_4, conv_layer_7, conv_layer_10, conv_layer_13],
                                            ["conv_2", "conv_4", "conv_7", "conv_10", "conv_13"],
                                            [pass_layer_1, pass_layer_2, pass_layer_3, pass_layer_4, pass_layer_5],
                                            ["pass_1", "pass_2", "pass_3", "pass_4", "pass_5"]):
        syn_avg = AvgPool2dConnection(source=src, target=trg, 
                                      kernel_size=2, stride=2)
        bn_model.add_connection(connection=syn_avg, source=src_name, target=trg_name)

    # Make dense connections
    for src, src_name, trg, trg_name, w in zip([pass_layer_5, dense_layer_1, dense_layer_2],
                                               ["pass_5", "dense_1", "dense_2"],
                                               [dense_layer_1, dense_layer_2,  dense_layer_3],
                                               ["dense_1", "dense_2", "dense_3"],
                                               [syn_dense_1_w, tf_model.get_weights()[14], tf_model.get_weights()[15]]):
        syn_dense = bnn.topology.Connection(
            source=src, target=trg,
            w=torch.tensor(w))
        bn_model.add_connection(connection=syn_dense, source=src_name, target=trg_name)
        assert (syn_dense.w.numpy() == w).all()
        assert (syn_dense.b == 0.0).all()

    out_monitor = bnn.monitors.Monitor(obj=dense_layer_3, state_vars=('s'), time=classify_time)
    bn_model.add_monitor(monitor=out_monitor, name='dense_3')

    bn_model.to('cuda')

    class Dataset(torch.utils.data.dataset.TensorDataset):

        def __init__(self, x, y, x_encoder=None, y_encoder=None):
            self.x = torch.Tensor(x)
            self.y = torch.Tensor(y)
            if x_encoder is None:
                x_encoder = NullEncoder()
            if y_encoder is None:
                y_encoder = NullEncoder()
            self.x_encoder = x_encoder
            self.y_encoder = y_encoder

        def __getitem__(self, i):
            return {
                "x": self.x[i],
                "y": self.y[i],
                "encoded_x": self.x_encoder(self.x[i]),
                "encoded_y": self.y_encoder(self.y[i]),
            }

        def __len__(self):
            return len(self.x)

    x_test_chan_first *= 1000 # bindsnet rates are in Hertz, not spikes per msec
    dataset = Dataset(x_test_chan_first, y_test, x_encoder=PoissonEncoder(time=classify_time, dt=dt))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    a = t.perf_counter()

    n_correct = 0
    for i, batch in tqdm(enumerate(dataloader)):

        inputs = {'input': batch['encoded_x'].to('cuda')}
        bn_model.run(inputs=inputs, time=classify_time)

        spikes = out_monitor.get('s').to('cpu')
        spikes_count = spikes.sum(0)
        predictions = spikes_count.argmax(1)
        targets = batch['encoded_y'][0, :, 0]
        for prediction, target in zip(predictions, targets):
            n_correct += prediction == target

        bn_model.reset_state_variables()

    accuracy = n_correct.to(torch.float) / x_test_chan_first.shape[0] * 100

    b = t.perf_counter()

    print('BindsNET accuracy: {}%'.format(accuracy))
    print('batch_size  classify_time  clock_time')
    print('{}  {}  {}'.format(batch_size, classify_time, b - a))
