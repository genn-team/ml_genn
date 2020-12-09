import tensorflow as tf
from tensorflow.keras import (models, layers, datasets, callbacks, optimizers,
                              initializers, regularizers)
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensor_genn import Model, InputType
from tensor_genn.norm import DataNorm, SpikeNorm
from tensor_genn.utils import parse_arguments, raster_plot
import numpy as np

# Learning rate schedule
def schedule(epoch, learning_rate):
    if epoch < 81:
        return 0.05
    elif epoch < 122:
        return 0.005
    else:
        return 0.0005

def initializer(shape, dtype=None):
    stddev = np.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)

if __name__ == '__main__':
    args = parse_arguments('VGG16 classifier model')
    print('arguments: ' + str(vars(args)))

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Retrieve and normalise CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train[:args.n_train_samples] / 255.0
    x_train -= np.average(x_train)
    y_train = y_train[:args.n_train_samples, 0]

    x_test = x_test[:args.n_test_samples] / 255.0
    x_test -= np.average(x_test)
    y_test = y_test[:args.n_test_samples, 0]
    x_norm = x_train[np.random.choice(x_train.shape[0], args.n_norm_samples, replace=False)]

    # Check input size
    if x_train.shape[1] < 32 or x_train.shape[2] < 32:
        raise ValueError('input must be at least 32x32')

    # If we should augment training data
    if args.augment_training:
        # Create image data generator
        data_gen = ImageDataGenerator(horizontal_flip=True)
        
        # Get training iterator
        iter_train = data_gen.flow(x_train, y_train, batch_size=256)
    
    # Create L2 regularizer
    regularizer = regularizers.l2(0.0001)
    
    # Create, train and evaluate TensorFlow model
    tf_model = models.Sequential([
        layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, input_shape=x_train.shape[1:], 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.3),
        layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False, 
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

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
    ], name='vgg16')

    if args.reuse_tf_model:
        with CustomObjectScope({'initializer': initializer}):
            tf_model = models.load_model('vgg16_tf_model')
    else:
        callbacks = [callbacks.LearningRateScheduler(schedule)]
        if args.record_tensorboard:
            callbacks.append(callbacks.TensorBoard(log_dir="logs", histogram_freq=1))

        optimizer = optimizers.SGD(lr=0.05, momentum=0.9)

        tf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        if args.augment_training:
            steps_per_epoch = x_train.shape[0] // 256
            tf_model.fit(iter_train, steps_per_epoch=steps_per_epoch, epochs=200, callbacks=callbacks)
        else:
            tf_model.fit(x_train, y_train, batch_size=256, epochs=200, shuffle=True, callbacks=callbacks)
        
        models.save_model(tf_model, 'vgg16_tf_model', save_format='h5')
    tf_model.evaluate(x_test, y_test)

    # Create, normalise and evaluate TensorGeNN model
    tg_model = Model.convert_tf_model(tf_model, input_type=args.input_type, synapse_type=args.synapse_type)
    tg_model.compile(dt=args.dt, rng_seed=args.rng_seed, batch_size=args.batch_size, share_weights=args.share_weights)

    if args.norm_method == 'data-norm':
        norm = DataNorm([x_norm], tf_model)
        norm.normalize(tg_model)
    elif args.norm_method == 'spike-norm':
        norm = SpikeNorm([x_norm])
        norm.normalize(tg_model, args.classify_time)

    acc, spk_i, spk_t = tg_model.evaluate([x_test], [y_test], args.classify_time, save_samples=args.save_samples)

    # Report TensorGeNN model results
    print('Accuracy of VGG16 GeNN model: {}%'.format(acc[0]))
    if args.plot:
        names = ['input_nrn'] + [name + '_nrn' for name in tg_model.layer_names]
        neurons = [tg_model.g_model.neuron_populations[name] for name in names]
        raster_plot(spk_i, spk_t, neurons)