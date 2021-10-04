import tensorflow as tf
from ml_genn import Model
from ml_genn.converters import DataNorm, SpikeNorm, FewSpike, Simple
from ml_genn.utils import parse_arguments, raster_plot
import numpy as np
from six import iteritems
from time import perf_counter
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras import (callbacks, models, optimizers, regularizers)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ml_genn.utils import parse_arguments, raster_plot
from tensorflow.keras.utils import model_to_dot

"""
def initializer(shape, dtype=None):
    stddev = np.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)

def initializer_res(shape, dtype=None):
    stddev = np.sqrt(2.0) / float(shape[0] * shape[1] * shape[3])
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)
"""
initializer = "he_normal"
initializer_res = "he_normal"

def resnetFunctionalIdentity(input_layer, filters, identity_cnn=False):
    stride = 2 if identity_cnn else 1
    input_layer = tf.keras.layers.ReLU()(input_layer)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', strides=stride, kernel_initializer=initializer_res, use_bias=False, activation='relu')(input_layer)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', strides=1, kernel_initializer=initializer_res, use_bias=False)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    if identity_cnn:
        res = tf.keras.layers.Conv2D(filters, kernel_size=(1,1), padding='same', strides=2, kernel_initializer=initializer_res, use_bias=False)(input_layer) 
    else:
        res = input_layer

    x = tf.keras.layers.Add()([x, res])
    return x

def resnet18(num_classes):
    input = tf.keras.layers.Input(shape=(32,32,3))
    x = tf.keras.layers.Conv2D(64, 7, padding='same', strides=2, kernel_initializer=initializer, activation='relu', use_bias=False)(input)
    x = tf.keras.layers.AveragePooling2D(3, strides=2)(x)

    x = resnetFunctionalIdentity(x, 64)
    x = resnetFunctionalIdentity(x, 64)
    
    x = resnetFunctionalIdentity(x, 128, True)
    x = resnetFunctionalIdentity(x, 128)
    
    x = resnetFunctionalIdentity(x, 256, True)
    x = resnetFunctionalIdentity(x, 256)
    
    x = resnetFunctionalIdentity(x, 512, True)
    x = resnetFunctionalIdentity(x, 512)

    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False)(x)
    model = tf.keras.Model(inputs=input, outputs=x, name='Resnet18')

    return model

if __name__ == '__main__':
    args = parse_arguments('Resnet classifier')
    print('arguments: ' + str(vars(args)))

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Retrieve and normalise CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train[:args.n_train_samples] / 255.0
    x_train -= np.average(x_train)
    y_train = y_train[:args.n_train_samples, 0]

    x_test = x_test[:args.n_test_samples] / 255.0
    x_test -= np.average(x_test)
    y_test = y_test[:args.n_test_samples, 0]
    x_norm = x_train[np.random.choice(x_train.shape[0], args.n_norm_samples, replace=False)]

    batch_size = 256
    steps_per_epoch = x_train.shape[0] // batch_size

    # Check input size
    if x_train.shape[1] < 32 or x_train.shape[2] < 32:
        raise ValueError('input must be at least 32x32')

    # Create image data generator
    data_gen = ImageDataGenerator(horizontal_flip=True, validation_split=0.1)

    # Get training iterator
    iter_train = data_gen.flow(x_train, y_train, batch_size=batch_size, subset="training")
    iter_validate = data_gen.flow(x_train, y_train, batch_size=batch_size, subset="validation")

    # Creating the model
    tf_model = resnet18(10)
    tf_model.build(input_shape = (None,32,32,3))

    if args.reuse_tf_model:
        tf_model.load_weights('resnet_tf_weights.h5')
    else:
        # Configure matching schedules for learning rate and weight decay schedules
        steps = [81 * steps_per_epoch, 122 * steps_per_epoch]
        decay = [1.0, 0.1, 0.01]
        lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            steps, [0.05 * d for d in decay])
        wd_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            steps, [0.0001 * d for d in decay])
        
        optimizer = SGDW(learning_rate=lr_schedule, momentum=0.9, nesterov=True, weight_decay=wd_schedule)
        tf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        fit_callbacks = []#,
#                        callbacks.EarlyStopping(patience=10)]
    
        if args.record_tensorboard:
            fit_callbacks.append(callbacks.TensorBoard(log_dir="logs", histogram_freq=1))

        tf_model.fit(iter_train, validation_data=iter_validate, epochs=200, callbacks=fit_callbacks)
        
        tf_model.save_weights('resnet_tf_weights.h5')

    tf_eval_start_time = perf_counter()
    tf_model.evaluate(x_test, y_test)
    print("TF evaluation:%f" % (perf_counter() - tf_eval_start_time))

    # Create, suitable converter to convert TF model to ML GeNN
    converter = args.build_converter(x_norm, signed_input=True, K=10, norm_time=2500)

    # Convert and compile ML GeNN model
    mlg_model = Model.convert_tf_model(tf_model, converter=converter, connectivity_type=args.connectivity_type,
        input_type=args.input_type, dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed,
        kernel_profiling=args.kernel_profiling)

    time = 10 if args.converter == 'few-spike' else 2500
    mlg_eval_start_time = perf_counter()
    acc, spk_i, spk_t = mlg_model.evaluate([x_test], [y_test], time, save_samples=args.save_samples)
    print("MLG evaluation:%f" % (perf_counter() - mlg_eval_start_time))

    if args.kernel_profiling:
        print("Kernel profiling:")
        for n, t in iteritems(mlg_model.get_kernel_times()):
            print("\t%s: %fs" % (n, t))

    # Report ML GeNN model results
    print('Accuracy of Resnet18 GeNN model: {}%'.format(acc[0]))
    if args.plot:
        neurons = [l.neurons.nrn for l in mlg_model.layers]
        raster_plot(spk_i, spk_t, neurons, time=time)
