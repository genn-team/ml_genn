import tensorflow as tf
from tensorflow.keras import models, layers, datasets
from tensor_genn import TGModel, InputType
from tensor_genn.norm import DataNorm, SpikeNorm
from tensor_genn.utils import parse_arguments, raster_plot
import numpy as np

class SimpleCNN(TGModel):
    def __init__(self, x_train, y_train, batch_size=1, dt=1.0, input_type=InputType.IF, rate_factor=1.0, rng_seed=0):
        super(SimpleCNN, self).__init__()

        # Define TensorFlow model
        tf_model = models.Sequential([
            layers.Conv2D(16, 5, padding='valid', activation='relu', use_bias=False, input_shape=x_train.shape[1:]),
            layers.AveragePooling2D(2),
            layers.Conv2D(8, 5, padding='valid', activation='relu', use_bias=False),
            layers.AveragePooling2D(2),
            layers.Flatten(),
            layers.Dense(128, activation='relu', use_bias=False),
            layers.Dense(64, activation='relu', use_bias=False),
            layers.Dense(y_train.max() + 1, activation='softmax', use_bias=False),
        ], name='simple_cnn')

        # Train and convert model
        #tf_model = models.load_model('simple_cnn_tf_model')
        tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        tf_model.fit(x_train, y_train, epochs=10)
        #models.save_model(tf_model, 'simple_cnn_tf_model', save_format='h5')
        self.convert_tf_model(tf_model)
        self.compile(batch_size=batch_size, dt=dt, input_type=input_type, rate_factor=rate_factor, rng_seed=rng_seed)
        self.tf_model = tf_model


if __name__ == '__main__':
    args = parse_arguments('Simple CNN classifier model')
    print('arguments: ' + str(vars(args)))

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Retrieve and normalise MNIST dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train[:args.n_train_samples].reshape((-1, 28, 28, 1)) / 255.0
    y_train = y_train[:args.n_train_samples]
    x_test = x_test[:args.n_test_samples].reshape((-1, 28, 28, 1)) / 255.0
    y_test = y_test[:args.n_test_samples]
    x_norm = x_train[np.random.choice(x_train.shape[0], args.n_norm_samples, replace=False)]

    # Create, normalise and evaluate TensorGeNN model
    tg_model = SimpleCNN(x_train, y_train,
                         batch_size=args.batch_size, dt=args.dt, input_type=args.input_type,
                         rate_factor=args.rate_factor, rng_seed=args.rng_seed)
    tg_model.tf_model.evaluate(x_test, y_test)
    if args.norm_method == 'data-norm':
        norm = DataNorm(x_norm, batch_size=None)
        norm.normalize(tg_model)
    elif args.norm_method == 'spike-norm':
        norm = SpikeNorm(x_norm)
        norm.normalize(tg_model, args.classify_time)
    acc, spk_i, spk_t = tg_model.evaluate(x_test, y_test, args.classify_time,
                                          save_samples=args.save_samples)

    # Report TensorGeNN model results
    print('Accuracy of SimpleCNN GeNN model: {}%'.format(acc))
    if args.plot:
        names = ['input_nrn'] + [name + '_nrn' for name in tg_model.layer_names]
        neurons = [tg_model.g_model.neuron_populations[name] for name in names]
        raster_plot(spk_i, spk_t, neurons)
