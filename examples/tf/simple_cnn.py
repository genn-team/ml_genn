import tensorflow as tf
import numpy as np

from tensorflow.keras import models, layers, datasets

from ml_genn.callbacks import SpikeRecorder
from ml_genn_tf.converters.spike_norm import spike_normalise
from arguments import parse_arguments
from plotting import plot_spikes
from time import perf_counter


args = parse_arguments('Simple CNN classifier model')
print('arguments: ' + str(vars(args)))

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Load MNIST data
(train_x, train_y), (validate_x, validate_y) = datasets.mnist.load_data()
train_x = train_x[:args.n_train_samples].reshape((-1, 28, 28, 1)) / 255.0
train_y = train_y[:args.n_train_samples]
validate_x = validate_x[:args.n_test_samples].reshape((-1, 28, 28, 1)) / 255.0
validate_y = validate_y[:args.n_test_samples]

# ML GeNN norm dataset
norm_i = np.random.choice(train_x.shape[0], args.n_norm_samples, replace=False)
mlg_norm_ds = tf.data.Dataset.from_tensor_slices((train_x[norm_i], train_y[norm_i]))
mlg_norm_ds = mlg_norm_ds.batch(args.batch_size)
mlg_norm_ds = mlg_norm_ds.prefetch(tf.data.AUTOTUNE)

# Create and compile TF model
tf_model = models.Sequential([
    layers.Conv2D(16, 5, padding='valid', activation='relu', use_bias=False, input_shape=train_x.shape[1:]),
    layers.AveragePooling2D(2),
    layers.Conv2D(8, 5, padding='valid', activation='relu', use_bias=False),
    layers.AveragePooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu', use_bias=False),
    layers.Dense(64, activation='relu', use_bias=False),
    layers.Dense(train_y.max() + 1, activation='softmax', use_bias=False),
], name='simple_cnn')
tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if args.reuse_tf_model:
    # Load old weights
    tf_model.load_weights('simple_cnn_tf_weights.h5')

else:
    # Fit TF model
    tf_model.fit(train_x, train_y, epochs=10)

    # Save weights
    tf_model.save_weights('simple_cnn_tf_weights.h5')

# Evaluate TF model
tf_eval_start_time = perf_counter()
tf_model.evaluate(validate_x, validate_y)
print("TF evaluation time: %f" % (perf_counter() - tf_eval_start_time))

# Create a suitable converter to convert TF model to ML GeNN
K = 8
T = 500
converter = args.build_converter(mlg_norm_ds, signed_input=False, 
                                 k=K, evaluate_timesteps=T)

# Convert and compile ML GeNN model
net, net_inputs, net_outputs, tf_layer_pops = converter.convert(tf_model)

if args.converter == "spike-norm":
    spike_normalise(net, net_inputs, net_outputs, mlg_norm_ds, evaluate_timesteps=T,
                    num_batches=int(np.ceil(args.n_norm_samples / args.batch_size)),
                    dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed,
                    kernel_profiling=args.kernel_profiling)

# If we should plot any spikes, turn on spike recording for all populations
if len(args.plot_sample_spikes) > 0:
    for l in tf_model.layers:
        if l in tf_layer_pops:
            tf_layer_pops[l].record_spikes = True
    
# Create suitable compiler for model
compiler = converter.create_compiler(prefer_in_memory_connect=args.prefer_in_memory_connect,
                                     dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed, 
                                     kernel_profiling=args.kernel_profiling)
compiled_net = compiler.compile(net, inputs=net_inputs, outputs=net_outputs)
compiled_net.genn_model.timing_enabled = args.kernel_profiling
    
with compiled_net:
    callbacks = ["batch_progress_bar"]

    # If we should plot any spikes, add spike recorder callback for all populations
    if len(args.plot_sample_spikes) > 0:
        for l in tf_model.layers:
            if l in tf_layer_pops:
                callbacks.append(
                    SpikeRecorder(tf_layer_pops[l], key=l.name,
                                  example_filter=args.plot_sample_spikes))

    # Evaluate ML GeNN modelplot_spikes
    start_time = perf_counter()
    metrics, cb_data = compiled_net.evaluate(
        {net_inputs[0]: validate_x}, {net_outputs[0]: validate_y},
        callbacks=callbacks)
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
        
