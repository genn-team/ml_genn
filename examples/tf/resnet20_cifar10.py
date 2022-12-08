from six import iteritems
from time import perf_counter
import numpy as np
import tensorflow as tf

from arguments import parse_arguments
from cifar10_dataset import *

tf_batch_size = 256
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

def resnet20():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False,
                               kernel_initializer=init_non_residual, kernel_regularizer=regu)(inputs)
    x = tf.keras.layers.ReLU()(x)

    # Sengupta et al (2019) use two extra plain pre-processing layers,
    # so this technically isn't ResNet20.
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
    x = tf.keras.layers.Dense(10, use_bias=False)(x)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='resnet20_cifar10')

    return model


if __name__ == '__main__':
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    args = parse_arguments('ResNet20 classifier')
    print('arguments: ' + str(vars(args)))

    # training dataset
    train_ds = cifar10_dataset_train()
    tf_train_ds = train_ds.batch(tf_batch_size)
    tf_train_ds = tf_train_ds.prefetch(tf.data.AUTOTUNE)

    # validation dataset
    validate_ds = cifar10_dataset_validate()
    tf_validate_ds = validate_ds.batch(tf_batch_size)
    tf_validate_ds = tf_validate_ds.prefetch(tf.data.AUTOTUNE)

    # ML GeNN norm dataset
    mlg_norm_ds = train_ds.take(args.n_norm_samples)
    mlg_norm_ds = mlg_norm_ds.batch(args.batch_size)
    mlg_norm_ds = mlg_norm_ds.prefetch(tf.data.AUTOTUNE)

    # ML GeNN validation dataset
    if args.n_test_samples is None:
        args.n_test_samples = 10000
    mlg_validate_ds = validate_ds.take(args.n_test_samples)
    mlg_validate_ds = mlg_validate_ds.map(lambda x, y: (x, y[0]), num_parallel_calls=tf.data.AUTOTUNE)
    mlg_validate_ds = mlg_validate_ds.batch(args.batch_size)
    mlg_validate_ds = mlg_validate_ds.prefetch(tf.data.AUTOTUNE)

    # Create and compile TF model
    tf_model = resnet20()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    tf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if args.reuse_tf_model:
        # Load old weights
        tf_model.load_weights('resnet20_cifar10_tf_weights.h5')

    else:
        # Fit TF model
        callbacks = []
        if args.record_tensorboard:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1))
        tf_model.fit(tf_train_ds, validation_data=tf_validate_ds, epochs=epochs, callbacks=callbacks)

        # Save weights
        tf_model.save_weights('resnet20_cifar10_tf_weights.h5')

    # Evaluate TF model
    tf_eval_start_time = perf_counter()
    tf_model.evaluate(tf_validate_ds)
    print("TF evaluation time: %f" % (perf_counter() - tf_eval_start_time))

    # Create a suitable converter to convert TF model to ML GeNN
    K = 8
    T = 1000
    converter = args.build_converter(mlg_norm_ds, signed_input=True, 
                                     k=K, evaluate_timesteps=T)

    # Convert and compile ML GeNN model
    mlg_net, mlg_net_inputs, mlg_net_outputs = converter.convert(tf_model)

    # Create suitable compiler for model
    compiler = converter.create_compiler(prefer_in_memory_connect=args.prefer_in_memory_connect,
                                         dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed, 
                                         kernel_profiling=args.kernel_profiling)
    compiled_net = compiler.compile(mlg_net, inputs=mlg_net_inputs, 
                                    outputs=mlg_net_outputs)
    compiled_net.genn_model.timing_enabled = args.kernel_profiling
        
    with compiled_net:
        # Evaluate ML GeNN model
        start_time = perf_counter()
        num_batches = args.n_test_samples // args.batch_size
        accuracy = compiled_net.evaluate_batch_iter(mlg_net_inputs, 
                                                    mlg_net_outputs,
                                                    iter(mlg_validate_ds),
                                                    num_batches=num_batches)
        end_time = perf_counter()
        print(f"Accuracy = {100.0 * accuracy[mlg_net_outputs[0]].result}%")
        print(f"Time = {end_time - start_time} s")
        
        if args.kernel_profiling:
            reset_time = compiled_net.genn_model.get_custom_update_time("Reset")
            print(f"Kernel profiling:\n"
                  f"\tinit_time: {compiled_net.genn_model.init_time} s\n"
                  f"\tinit_sparse_time: {compiled_net.genn_model.init_sparse_time} s\n"
                  f"\tneuron_update_time: {compiled_net.genn_model.neuron_update_time} s\n"
                  f"\tpresynaptic_update_time: {compiled_net.genn_model.presynaptic_update_time} s\n"
                  f"\tcustom_update_reset_time: {reset_time} s")
