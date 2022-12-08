import numpy as np
import tensorflow as tf

from ml_genn.callbacks import SpikeRecorder

from six import iteritems
from time import perf_counter
from arguments import parse_arguments
from plotting import plot_spikes

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

def init(shape, dtype=None):
    stddev = tf.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)

def vgg16_block(x, n, filters, downsample=2, dropout_rate=dropout_rate):
    for i in range(n):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False,
                                   kernel_initializer=init, kernel_regularizer=regu)(x)
        x = tf.keras.layers.ReLU()(x)
        if i < (n - 1):
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        else:
            x = tf.keras.layers.AveragePooling2D(downsample)(x)
    return x

def vgg16():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    x = vgg16_block(inputs, 2, 64, downsample=2, dropout_rate=dropout_rate)
    x = vgg16_block(x, 2, 128, downsample=2, dropout_rate=dropout_rate)
    x = vgg16_block(x, 3, 256, downsample=2, dropout_rate=dropout_rate)
    x = vgg16_block(x, 3, 512, downsample=2, dropout_rate=dropout_rate)
    x = vgg16_block(x, 3, 512, downsample=2, dropout_rate=dropout_rate)

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
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    args = parse_arguments('VGG16 classifier')
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

    # Create a suitable converter to convert TF model to ML GeNN
    K = 10
    T = 2500
    converter = args.build_converter(mlg_norm_ds, signed_input=True, 
                                     k=K, evaluate_timesteps=T)

    # Convert and compile ML GeNN model
    net, net_inputs, net_outputs, tf_layer_pops = converter.convert(tf_model)
    
    # If we should plot any spikes, turn on spike recording for all populations
    if len(args.plot_sample_spikes) > 0:
        for l in tf_model.layers:
            if l in tf_layer_pops:
                callbacks.append(
                    SpikeRecorder(tf_layer_pops[l], key=l.name,
                                  example_filter=args.plot_sample_spikes))

    # Create suitable compiler for model
    compiler = converter.create_compiler(prefer_in_memory_connect=args.prefer_in_memory_connect,
                                         dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed, 
                                         kernel_profiling=args.kernel_profiling)
    compiled_net = compiler.compile(net, inputs=net_inputs, 
                                    outputs=net_outputs)
    compiled_net.genn_model.timing_enabled = args.kernel_profiling
    
    with compiled_net:
        # If we should plot any spikes, add spike recorder callback for all populations
        callbacks = ["batch_progress_bar"]
        if len(args.plot_sample_spikes) > 0:
            for p in net.populations:
                callbacks.append(SpikeRecorder(p, example_filter=args.plot_sample_spikes))
                
        # Evaluate ML GeNN model
        start_time = perf_counter()
        num_batches = args.n_test_samples // args.batch_size
        metrics, cb_data = compiled_net.evaluate_batch_iter(
            net_inputs, net_outputs, iter(mlg_validate_ds),
            num_batches=num_batches, callbacks=callbacks)
        end_time = perf_counter()
        print(f"Accuracy = {100.0 * metrics[net_outputs[0]].result}%")
        print(f"Time = {end_time - start_time} s")
        
        if args.kernel_profiling:
            reset_time = compiled_net.genn_model.get_custom_update_time("Reset")
            print(f"Kernel profiling:\n"
                  f"\tinit_time: {compiled_net.genn_model.init_time} s\n"
                  f"\tinit_sparse_time: {compiled_net.genn_model.init_sparse_time} s\n"
                  f"\tneuron_update_time: {compiled_net.genn_model.neuron_update_time} s\n"
                  f"\tpresynaptic_update_time: {compiled_net.genn_model.presynaptic_update_time} s\n"
                  f"\tcustom_update_reset_time: {reset_time} s")
        
        # Plot spikes if desired
        if len(args.plot_sample_spikes) > 0:
            plot_spikes(cb_data, args.plot_sample_spikes)
