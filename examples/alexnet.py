import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    print(tf.config.experimental.get_memory_growth(gpu))

from tensorflow.keras import (models, layers, datasets, callbacks, optimizers,
                              initializers, regularizers)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from six import iteritems
from time import perf_counter
from ml_genn import Model
from ml_genn.utils import parse_arguments, raster_plot


# Learning rate schedule
def schedule(epoch, learning_rate):
    if epoch < 81:
        return 0.05
    elif epoch < 122:
        return 0.005
    else:
        return 0.0005

if __name__ == '__main__':
    args = parse_arguments('AlexNet classifier model')

    n_norm_samples=1000
    #Load Dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    #Normalize data
    x_train = x_train / 255.0
    x_train -= np.average(x_train)
    y_train = y_train[:args.n_train_samples, 0]

    x_test = x_test / 255.0
    x_test -= np.average(x_test)
    y_test = y_test[:args.n_test_samples, 0]

    index_norm=np.random.choice(x_train.shape[0], n_norm_samples, replace=False)
    x_norm = x_train[index_norm]
    y_norm = y_train[index_norm]

    # If we should augment training data
    if args.augment_training:
        # Create image data generator
        data_gen = ImageDataGenerator(horizontal_flip=True, validation_split=0.1)

        # Get training and validation iterators
        iter_train = data_gen.flow(x_train, y_train, batch_size=256, subset="training")
        iter_validate = data_gen.flow(x_train, y_train, batch_size=256, subset="validation")

    # Create L2 regularizer
    regularizer = regularizers.l2(0.0001)
    initializer = "he_uniform"

    tf_model = models.Sequential([
        layers.Conv2D(filters=96,kernel_size=(11,11), padding='same', activation='relu', use_bias=False,
        kernel_initializer=initializer, kernel_regularizer=regularizer,input_shape=x_train.shape[1:]),
        layers.AveragePooling2D(2),
        layers.Conv2D(filters=256, kernel_size=(5,5),  activation='relu', padding="same", use_bias=False,
        kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),
        layers.Conv2D(filters=384, kernel_size=(3,3),  activation='relu', padding="same", use_bias=False,
        kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu', padding="same", use_bias=False,
        kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding="same", use_bias=False,
        kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu', use_bias=False, kernel_regularizer=regularizer),
        layers.Dense(4096, activation='relu', use_bias=False, kernel_regularizer=regularizer),
        layers.Dense(y_train.max() + 1, activation='softmax', use_bias=False, kernel_regularizer=regularizer)
    ],name="alexnet")

    if args.reuse_tf_model:
        tf_model = models.load_model('alexnet_tf_model')
    else:
        fit_callbacks = [callbacks.LearningRateScheduler(schedule),
                         callbacks.EarlyStopping(patience=4)]
        if args.record_tensorboard:
            fit_callbacks.append(callbacks.TensorBoard(log_dir="logs", histogram_freq=1))

        optimizer = optimizers.SGD(lr=0.05, momentum=0.9)
        tf_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        #train TensorFlow model
        if args.augment_training:
            tf_model.fit(iter_train, epochs=75, validation_data=iter_validate, callbacks=fit_callbacks)
        else:
            tf_model.fit(x_train, y_train, batch_size=256, epochs=75, shuffle=True, 
                         validation_split=0.1, callbacks=fit_callbacks)

        #Save alexnet_tf_model
        models.save_model(tf_model, 'alexnet_tf_model', save_format='h5')

    tf_eval_start_time = perf_counter()
    tf_model.evaluate(x_test, y_test)
    print("TF evaluation:%f" % (perf_counter() - tf_eval_start_time))

    # Create a suitable converter to convert TF model to ML GeNN
    converter = args.build_converter(x_norm, signed_input=True, K=10, norm_time=500)

    # Convert and compile ML GeNN model
    mlg_model = Model.convert_tf_model(
        tf_model, converter=converter, connectivity_type=args.connectivity_type,
        input_type=args.input_type, dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed, 
        kernel_profiling=args.kernel_profiling)

    time = 10 if args.converter == 'few-spike' else 500
    mlg_eval_start_time = perf_counter()
    acc, spk_i, spk_t = mlg_model.evaluate([x_test], [y_test], time, save_samples=args.save_samples)
    print("MLG evaluation:%f" % (perf_counter() - mlg_eval_start_time))

    if args.kernel_profiling:
        print("Kernel profiling:")
        for n, t in iteritems(mlg_model.get_kernel_times()):
            print("\t%s: %fs" % (n, t))

    # Report ML GeNN model results
    print('Accuracy of AlexNet GeNN model: {}%'.format(acc[0]))
    if args.plot:
        neurons = [l.neurons.nrn for l in mlg_model.layers]
        raster_plot(spk_i, spk_t, neurons, time=time)
