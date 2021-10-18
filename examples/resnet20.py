from six import iteritems
from time import perf_counter
import numpy as np
import tensorflow as tf
from ml_genn import Model
from ml_genn.converters import DataNorm, SpikeNorm, FewSpike, Simple
from ml_genn.utils import parse_arguments, raster_plot


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


def init_non_residual(shape, dtype=None):
    stddev = tf.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)

def init_residual(shape, dtype=None):
    stddev = tf.sqrt(2.0) / float(shape[0] * shape[1] * shape[3])
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)


def resnet20_block(input_layer, filters, downsample=False):
    stride = 2 if downsample else 1

    x = tf.keras.layers.Conv2D(filters, 3, padding='same', strides=stride, use_bias=False,
                               kernel_initializer=init_residual, kernel_regularizer=regu)(input_layer)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', strides=1, use_bias=False,
                               kernel_initializer=init_residual, kernel_regularizer=regu)(x)

    if downsample:
        res = tf.keras.layers.Conv2D(filters, 1, padding='same', strides=2, use_bias=False,
                                     kernel_initializer='zeros', kernel_regularizer=regu)(input_layer)
    else:
        res = input_layer

    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.ReLU()(x)

    return x


def resnet20(num_classes):
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False,
                               kernel_initializer=init_non_residual, kernel_regularizer=regu)(inputs)
    x = tf.keras.layers.ReLU()(x)

    # Sengupta et al use 'two extra plain pre-processing layers', so this technically isn't ResNet-20
    x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False,
                               kernel_initializer=init_non_residual, kernel_regularizer=regu)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False,
                               kernel_initializer=init_non_residual, kernel_regularizer=regu)(x)
    x = tf.keras.layers.ReLU()(x)

    x = resnet20_block(x, 16)
    x = resnet20_block(x, 16)
    x = resnet20_block(x, 16)

    x = resnet20_block(x, 32, True)
    x = resnet20_block(x, 32)
    x = resnet20_block(x, 32)

    x = resnet20_block(x, 64, True)
    x = resnet20_block(x, 64)
    x = resnet20_block(x, 64)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, use_bias=False)(x)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='ResNet20')

    return model


if __name__ == '__main__':
    args = parse_arguments('Resnet classifier')
    print('arguments: ' + str(vars(args)))

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # load CIFAR-10 data
    data = tf.keras.datasets.cifar10.load_data()
    x_mean = np.array([125.3, 123.0, 113.9], dtype='float32')
    x_std = np.array([63.0, 62.1, 66.7], dtype='float32')
    (train_x, train_y), (validate_x, validate_y) = data
    train_x = train_x.astype('float32')
    validate_x = validate_x.astype('float32')

    def color_normalize(x, y, x_mean, x_std):
        x = (x - x_mean) / x_std
        return x, y

    def random_crop(x, y, size, padding):
        pad_shape = [x.shape[0] + padding, x.shape[1] + padding]
        x = tf.image.resize_with_crop_or_pad(x, pad_shape[0], pad_shape[1])
        x = tf.image.random_crop(x, (size, size, 3))
        return x, y

    def horizontal_flip(x, y):
        x = tf.image.random_flip_left_right(x)
        return x, y

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.map(lambda x, y: color_normalize(x, y, x_mean, x_std), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: random_crop(x, y, 32, 4), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: horizontal_flip(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    validate_ds = tf.data.Dataset.from_tensor_slices((validate_x, validate_y))
    validate_ds = validate_ds.map(lambda x, y: color_normalize(x, y, x_mean, x_std), num_parallel_calls=tf.data.AUTOTUNE)
    validate_ds = validate_ds.batch(batch_size)
    validate_ds = validate_ds.prefetch(tf.data.AUTOTUNE)

    # Creating the model
    tf_model = resnet20(10)
    #tf_model.summary()

    # compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    tf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model
    if args.reuse_tf_model:
        tf_model.load_weights('resnet_tf_weights.h5')
    else:
        callbacks = []
        if args.record_tensorboard:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1))
        tf_model.fit(train_ds, validation_data=validate_ds, epochs=epochs, callbacks=callbacks)
        tf_model.save_weights('resnet_tf_weights.h5')

    # evaluate model
    tf_eval_start_time = perf_counter()
    tf_model.evaluate(validate_ds)
    print("TF evaluation time: %f" % (perf_counter() - tf_eval_start_time))

    # prepare data for ML GeNN
    train_x = (train_x - x_mean) / x_std
    train_y = train_y[:, 0]
    validate_x = (validate_x - x_mean) / x_std
    validate_y = validate_y[:, 0]
    norm_x = train_x[np.random.choice(train_x.shape[0], args.n_norm_samples, replace=False)]

    # Create, suitable converter to convert TF model to ML GeNN
    converter = args.build_converter(norm_x, signed_input=True, K=10, norm_time=2500)

    # Convert and compile ML GeNN model
    mlg_model = Model.convert_tf_model(tf_model, converter=converter, connectivity_type=args.connectivity_type,
        input_type=args.input_type, dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed,
        kernel_profiling=args.kernel_profiling)

    time = 10 if args.converter == 'few-spike' else 2500
    mlg_eval_start_time = perf_counter()
    acc, spk_i, spk_t = mlg_model.evaluate([validate_x], [validate_y], time, save_samples=args.save_samples)
    print("MLG evaluation time: %f" % (perf_counter() - mlg_eval_start_time))

    if args.kernel_profiling:
        print("Kernel profiling:")
        for n, t in iteritems(mlg_model.get_kernel_times()):
            print("\t%s: %fs" % (n, t))

    # Report ML GeNN model results
    print('Accuracy of ResNet20 GeNN model: {}%'.format(acc[0]))
    if args.plot:
        neurons = [l.neurons.nrn for l in mlg_model.layers]
        raster_plot(spk_i, spk_t, neurons, time=time)
