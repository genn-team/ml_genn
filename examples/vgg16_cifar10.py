from six import iteritems
from time import perf_counter
import numpy as np
import tensorflow as tf

from ml_genn import Model
from ml_genn.converters import DataNorm, SpikeNorm, FewSpike, Simple
from ml_genn.utils import parse_arguments, raster_plot
from transforms import *


batch_size = 256
epochs = 200
learning_rate = 0.05
momentum = 0.9

steps_per_epoch = 50000 // epochs
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [81 * steps_per_epoch, 122 * steps_per_epoch],
    [learning_rate, learning_rate / 10.0, learning_rate / 100.0])

dropout_rate = 0.25

weight_decay = 0.0001
regu = tf.keras.regularizers.L2(weight_decay)


def init(shape, dtype=None):
    stddev = tf.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)


def vgg16_layer(input_layer, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False,
                               kernel_initializer=init, kernel_regularizer=regu)(input_layer)
    x = tf.keras.layers.ReLU()(x)
    return x


def vgg16():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    x = vgg16_layer(inputs, 64)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = vgg16_layer(x, 64)
    x = tf.keras.layers.AveragePooling2D(2)(x)

    x = vgg16_layer(x, 128)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = vgg16_layer(x, 128)
    x = tf.keras.layers.AveragePooling2D(2)(x)

    x = vgg16_layer(x, 256)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = vgg16_layer(x, 256)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = vgg16_layer(x, 256)
    x = tf.keras.layers.AveragePooling2D(2)(x)

    x = vgg16_layer(x, 512)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = vgg16_layer(x, 512)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = vgg16_layer(x, 512)
    x = tf.keras.layers.AveragePooling2D(2)(x)

    x = vgg16_layer(x, 512)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = vgg16_layer(x, 512)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = vgg16_layer(x, 512)
    x = tf.keras.layers.AveragePooling2D(2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, use_bias=False, kernel_regularizer=regu)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(4096, use_bias=False, kernel_regularizer=regu)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(10, use_bias=False, kernel_regularizer=regu)(x)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='vgg16_cifar10')

    return model


if __name__ == '__main__':
    args = parse_arguments('VGG16 classifier')
    print('arguments: ' + str(vars(args)))

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load CIFAR-10 data
    data = tf.keras.datasets.cifar10.load_data()
    cifar10_mean = np.array([125.3, 123.0, 113.9], dtype='float32')
    cifar10_std = np.array([63.0, 62.1, 66.7], dtype='float32')
    (train_x, train_y), (validate_x, validate_y) = data
    train_x = train_x.astype('float32')
    validate_x = validate_x.astype('float32')
    validate_x = validate_x[:args.n_test_samples]
    validate_y = validate_y[:args.n_test_samples]

    color_normalize_fn = color_normalize(cifar10_mean, cifar10_std)
    random_crop_fn = random_crop(32, 4)
    horizontal_flip_fn = horizontal_flip()

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(random_crop_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(horizontal_flip_fn, num_parallel_calls=tf.data.AUTOTUNE)
    tf_train_ds = train_ds.batch(batch_size)
    tf_train_ds = tf_train_ds.prefetch(tf.data.AUTOTUNE)

    validate_ds = tf.data.Dataset.from_tensor_slices((validate_x, validate_y))
    validate_ds = validate_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    tf_validate_ds = validate_ds.batch(batch_size)
    tf_validate_ds = tf_validate_ds.prefetch(tf.data.AUTOTUNE)

    # Create and compile TF model
    tf_model = vgg16()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    tf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if args.reuse_tf_model:
        # Load old weights
        tf_model.load_weights('vgg16_cifar10_tf_weights.h5')

    else:
        # Fit TF model
        callbacks = []
        if args.record_tensorboard:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1))
        tf_model.fit(tf_train_ds, validation_data=tf_validate_ds, epochs=epochs, callbacks=callbacks)

        # Save weights
        tf_model.save_weights('vgg16_cifar10_tf_weights.h5')

    # Evaluate TF model
    tf_eval_start_time = perf_counter()
    tf_model.evaluate(tf_validate_ds)
    print("TF evaluation time: %f" % (perf_counter() - tf_eval_start_time))

    # Prepare data for ML GeNN
    if args.n_test_samples is None:
        args.n_test_samples = 10000
    mlg_norm_ds = train_ds.take(args.n_norm_samples).as_numpy_iterator()
    norm_x = np.array([d[0] for d in mlg_norm_ds])
    mlg_validate_ds = tf.data.Dataset.from_tensor_slices((validate_x, validate_y))
    mlg_validate_ds = mlg_validate_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_validate_ds = mlg_validate_ds.map(lambda x, y: (x, y[0]), num_parallel_calls=tf.data.AUTOTUNE)
    mlg_validate_ds = mlg_validate_ds.batch(args.batch_size)
    mlg_validate_ds = mlg_validate_ds.as_numpy_iterator()

    # Create a suitable converter to convert TF model to ML GeNN
    K = 10
    T = 2500
    converter = args.build_converter(norm_x, signed_input=True, K=K, norm_time=T)

    # Convert and compile ML GeNN model
    mlg_model = Model.convert_tf_model(
        tf_model, converter=converter, connectivity_type=args.connectivity_type,
        dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed, 
        kernel_profiling=args.kernel_profiling)
    
    # Evaluate ML GeNN model
    time = K if args.converter == 'few-spike' else T
    mlg_eval_start_time = perf_counter()
    acc, spk_i, spk_t = mlg_model.evaluate_iterator(
        mlg_validate_ds, args.n_test_samples, time, save_samples=args.save_samples)
    print("MLG evaluation time: %f" % (perf_counter() - mlg_eval_start_time))

    if len(args.save_samples) > 0:
        num_spikes = 0
        for sample_spikes in spk_i:
            for layer_spikes in sample_spikes:
               num_spikes += len(layer_spikes)
        print("Mean spikes per sample: %f" % (num_spikes / len(args.save_samples)))

    if args.kernel_profiling:
        print("Kernel profiling:")
        for n, t in iteritems(mlg_model.get_kernel_times()):
            print("\t%s: %fs" % (n, t))

    # Report ML GeNN model results
    print('Accuracy of VGG16 GeNN model: {}%'.format(acc[0]))
    if args.plot:
        neurons = [l.neurons.nrn for l in mlg_model.layers]
        raster_plot(spk_i, spk_t, neurons, time=time)
