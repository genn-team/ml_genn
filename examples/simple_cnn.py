import tensorflow as tf
from tensorflow.keras import models, layers, datasets
from ml_genn import Model
from ml_genn.utils import parse_arguments, raster_plot
import numpy as np
from six import iteritems
from time import perf_counter


if __name__ == '__main__':
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
    converter = args.build_converter(mlg_norm_ds, signed_input=False, K=K, norm_time=T)

    # Convert and compile ML GeNN model
    mlg_model = Model.convert_tf_model(
        tf_model, converter=converter, connectivity_type=args.connectivity_type,
        dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed, 
        kernel_profiling=args.kernel_profiling)

    # Evaluate ML GeNN model
    time = K if args.converter == 'few-spike' else T
    mlg_eval_start_time = perf_counter()
    acc, spk_i, spk_t = mlg_model.evaluate(
        validate_x, validate_y, time, save_samples=args.save_samples)
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
    print(f'Accuracy of SimpleCNN GeNN model: {acc[0]}%')
    if args.plot:
        neurons = [l.neurons.nrn for l in mlg_model.layers]
        raster_plot(spk_i, spk_t, neurons, time=time)
