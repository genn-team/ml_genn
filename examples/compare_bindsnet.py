from tqdm import tqdm
import time as t
import numpy as np

import tensorflow as tf
from tensorflow.keras import models, layers, datasets

from ml_genn import Model
from ml_genn.utils import parse_arguments

import torch
import bindsnet.network as bnn
from bindsnet.datasets import DataLoader
from bindsnet.encoding import NullEncoder, PoissonEncoder


if __name__ == '__main__':
    args = parse_arguments('VGG16 classifier model')
    print('arguments: ' + str(vars(args)))

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    test_mlg = True
    test_bindsnet = True
    classify_time = 500
    dt = args.dt
    batch_size = args.batch_size

    # Retrieve and normalise CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train[:args.n_train_samples] / 255.0
    y_train = y_train[:args.n_train_samples, 0]
    x_test = x_test[:args.n_test_samples] / 255.0
    y_test = y_test[:args.n_test_samples, 0]
    x_norm = x_train[np.random.choice(x_train.shape[0], args.n_norm_samples, replace=False)]


    # TensorFlow model
    print('TENSORFLOW MODEL')
    tf_model = models.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', use_bias=False, input_shape=x_train.shape[1:]),
        layers.Dropout(0.3),
        layers.Conv2D(16, 3, padding='same', activation='relu', use_bias=False),
        layers.Dropout(0.3),
        #layers.AveragePooling2D(2),
        #layers.Conv2D(16, 3, strides=2, padding='same', activation='relu', use_bias=False),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=False),
        layers.Dropout(0.4),
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=False),
        layers.Dropout(0.4),
        #layers.AveragePooling2D(2),
        #layers.Conv2D(32, 3, strides=2, padding="same", activation="relu", use_bias=False),

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=False),
        layers.Dropout(0.4),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=False),
        layers.Dropout(0.4),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=False),
        layers.Dropout(0.4),
        #layers.AveragePooling2D(2),
        #layers.Conv2D(64, 3, strides=2, padding="same", activation="relu", use_bias=False),

        layers.Flatten(),
        layers.Dense(4096, activation="relu", use_bias=False),
        layers.Dropout(0.5),
        layers.Dense(4096, activation="relu", use_bias=False),
        layers.Dropout(0.5),
        layers.Dense(y_train.max() + 1, activation="softmax", use_bias=False),
    ], name='compare_4')

    if args.reuse_tf_model:
        tf_model = models.load_model('compare_tf_model_4')
    else:
        tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        tf_model.fit(x_train, y_train, epochs=50)
        models.save_model(tf_model, 'compare_tf_model_4', save_format='h5')
    tf_model.summary()
    tf_model.evaluate(x_test, y_test)


    # ML GeNN model
    if test_mlg:
        print('ML GENN MODEL')

        # Create a suitable converter to convert TF model to ML GeNN
        converter = args.build_converter(x_norm, K=8, norm_time=classify_time)

        # Convert and compile ML GeNN model
        mlg_model = Model.convert_tf_model(
            tf_model, converter=converter, connectivity_type=args.connectivity_type,
            dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed, 
            kernel_profiling=args.kernel_profiling)

        time = 8 if args.converter == 'few-spike' else classify_time

        a = t.perf_counter()

        acc, spk_i, spk_t = mlg_model.evaluate([x_test], [y_test], classify_time,
                                               save_samples=args.save_samples)

        b = t.perf_counter()

        print('ML GeNN accuracy: {}%'.format(acc[0]))
        print('batch_size  classify_time  clock_time')
        print('{}  {}  {}'.format(batch_size, classify_time, b - a))


    # BindsNET model
    if test_bindsnet:
        print('BINDSNET MODEL')

        bn_model = bnn.Network()

        # convert input to pytorch channel first from tf channel last
        x_test_chan_first = np.moveaxis(x_test, 3, 1)

        # convert conv weights to pytorch format [out_ch, in_ch, h, w] from tf format [h, w, in_ch, out_ch]
        syn_conv_1_w = np.moveaxis(tf_model.get_weights()[0], (3, 2), (0, 1))
        syn_conv_2_w = np.moveaxis(tf_model.get_weights()[1], (3, 2), (0, 1))
        syn_conv_3_w = np.moveaxis(tf_model.get_weights()[2], (3, 2), (0, 1))
        syn_conv_4_w = np.moveaxis(tf_model.get_weights()[3], (3, 2), (0, 1))
        syn_conv_5_w = np.moveaxis(tf_model.get_weights()[4], (3, 2), (0, 1))
        syn_conv_6_w = np.moveaxis(tf_model.get_weights()[5], (3, 2), (0, 1))
        syn_conv_7_w = np.moveaxis(tf_model.get_weights()[6], (3, 2), (0, 1))

        # convert dense weights to handle pytorch input format (channels first)
        syn_dense_1_w = np.empty(tf_model.get_weights()[7].shape, dtype=tf_model.get_weights()[7].dtype)
        for i in range(64):
            syn_dense_1_w[(i * 32 * 32):((i + 1) * 32 * 32)] = tf_model.get_weights()[7][i::64]

        input_layer = bnn.nodes.Input(n=(3 * 32 * 32), shape=(3, 32, 32))
        bn_model.add_layer(layer=input_layer, name='input')

        conv_layer_1 = bnn.nodes.IFNodes(n=(16 * 32 * 32), shape=(16, 32, 32), thresh=1.0, reset=0.0, refac=0)
        bn_model.add_layer(layer=conv_layer_1, name='conv_1')

        conv_layer_2 = bnn.nodes.IFNodes(n=(16 * 32 * 32), shape=(16, 32, 32), thresh=1.0, reset=0.0, refac=0)
        bn_model.add_layer(layer=conv_layer_2, name='conv_2')

        conv_layer_3 = bnn.nodes.IFNodes(n=(32 * 32 * 32), shape=(32, 32, 32), thresh=1.0, reset=0.0, refac=0)
        bn_model.add_layer(layer=conv_layer_3, name='conv_3')

        conv_layer_4 = bnn.nodes.IFNodes(n=(32 * 32 * 32), shape=(32, 32, 32), thresh=1.0, reset=0.0, refac=0)
        bn_model.add_layer(layer=conv_layer_4, name='conv_4')

        conv_layer_5 = bnn.nodes.IFNodes(n=(64 * 32 * 32), shape=(64, 32, 32), thresh=1.0, reset=0.0, refac=0)
        bn_model.add_layer(layer=conv_layer_5, name='conv_5')

        conv_layer_6 = bnn.nodes.IFNodes(n=(64 * 32 * 32), shape=(64, 32, 32), thresh=1.0, reset=0.0, refac=0)
        bn_model.add_layer(layer=conv_layer_6, name='conv_6')

        conv_layer_7 = bnn.nodes.IFNodes(n=(64 * 32 * 32), shape=(64, 32, 32), thresh=1.0, reset=0.0, refac=0)
        bn_model.add_layer(layer=conv_layer_7, name='conv_7')

        dense_layer_1 = bnn.nodes.IFNodes(n=4096, thresh=1.0, reset=0.0, refac=0)
        bn_model.add_layer(layer=dense_layer_1, name='dense_1')

        dense_layer_2 = bnn.nodes.IFNodes(n=4096, thresh=1.0, reset=0.0, refac=0)
        bn_model.add_layer(layer=dense_layer_2, name='dense_2')

        dense_layer_3 = bnn.nodes.IFNodes(n=10, thresh=1.0, reset=0.0, refac=0)
        bn_model.add_layer(layer=dense_layer_3, name='dense_3')

        syn_conv_1 = bnn.topology.Conv2dConnection(
            source=input_layer,
            target=conv_layer_1,
            kernel_size=3,
            padding=1,
            w=torch.tensor(syn_conv_1_w)
        )
        bn_model.add_connection(connection=syn_conv_1, source='input', target='conv_1')
        assert (syn_conv_1.w.numpy() == syn_conv_1_w).all()
        assert (syn_conv_1.b == 0.0).all()

        syn_conv_2 = bnn.topology.Conv2dConnection(
            source=conv_layer_1,
            target=conv_layer_2,
            kernel_size=3,
            padding=1,
            w=torch.tensor(syn_conv_2_w)
        )
        bn_model.add_connection(connection=syn_conv_2, source='conv_1', target='conv_2')
        assert (syn_conv_2.w.numpy() == syn_conv_2_w).all()
        assert (syn_conv_2.b == 0.0).all()

        syn_conv_3 = bnn.topology.Conv2dConnection(
            source=conv_layer_2,
            target=conv_layer_3,
            kernel_size=3,
            padding=1,
            w=torch.tensor(syn_conv_3_w)
        )
        bn_model.add_connection(connection=syn_conv_3, source='conv_2', target='conv_3')
        assert (syn_conv_3.w.numpy() == syn_conv_3_w).all()
        assert (syn_conv_3.b == 0.0).all()

        syn_conv_4 = bnn.topology.Conv2dConnection(
            source=conv_layer_3,
            target=conv_layer_4,
            kernel_size=3,
            padding=1,
            w=torch.tensor(syn_conv_4_w)
        )
        bn_model.add_connection(connection=syn_conv_4, source='conv_3', target='conv_4')
        assert (syn_conv_4.w.numpy() == syn_conv_4_w).all()
        assert (syn_conv_4.b == 0.0).all()

        syn_conv_5 = bnn.topology.Conv2dConnection(
            source=conv_layer_4,
            target=conv_layer_5,
            kernel_size=3,
            padding=1,
            w=torch.tensor(syn_conv_5_w)
        )
        bn_model.add_connection(connection=syn_conv_5, source='conv_4', target='conv_5')
        assert (syn_conv_5.w.numpy() == syn_conv_5_w).all()
        assert (syn_conv_5.b == 0.0).all()

        syn_conv_6 = bnn.topology.Conv2dConnection(
            source=conv_layer_5,
            target=conv_layer_6,
            kernel_size=3,
            padding=1,
            w=torch.tensor(syn_conv_6_w)
        )
        bn_model.add_connection(connection=syn_conv_6, source='conv_5', target='conv_6')
        assert (syn_conv_6.w.numpy() == syn_conv_6_w).all()
        assert (syn_conv_6.b == 0.0).all()

        syn_conv_7 = bnn.topology.Conv2dConnection(
            source=conv_layer_6,
            target=conv_layer_7,
            kernel_size=3,
            padding=1,
            w=torch.tensor(syn_conv_7_w)
        )
        bn_model.add_connection(connection=syn_conv_7, source='conv_6', target='conv_7')
        assert (syn_conv_7.w.numpy() == syn_conv_7_w).all()
        assert (syn_conv_7.b == 0.0).all()

        syn_dense_1 = bnn.topology.Connection(
            source=conv_layer_7,
            target=dense_layer_1,
            w=torch.tensor(syn_dense_1_w)
        )
        bn_model.add_connection(connection=syn_dense_1, source='conv_7', target='dense_1')
        assert (syn_dense_1.w.numpy() == syn_dense_1_w).all()
        assert (syn_dense_1.b == 0.0).all()

        syn_dense_2 = bnn.topology.Connection(
            source=dense_layer_1,
            target=dense_layer_2,
            w=torch.tensor(tf_model.get_weights()[8])
        )
        bn_model.add_connection(connection=syn_dense_2, source='dense_1', target='dense_2')
        assert (syn_dense_2.w.numpy() == tf_model.get_weights()[8]).all()
        assert (syn_dense_2.b == 0.0).all()

        syn_dense_3 = bnn.topology.Connection(
            source=dense_layer_2,
            target=dense_layer_3,
            w=torch.tensor(tf_model.get_weights()[9])
        )
        bn_model.add_connection(connection=syn_dense_3, source='dense_2', target='dense_3')
        assert (syn_dense_3.w.numpy() == tf_model.get_weights()[9]).all()
        assert (syn_dense_3.b == 0.0).all()

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
