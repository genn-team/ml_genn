import tensorflow as tf
from tensorflow.keras import models, layers, datasets
from tensor_genn import TGModel, InputType
from tensor_genn.norm import DataNorm, SpikeNorm
from tensor_genn.utils import parse_arguments, raster_plot
import numpy as np

class VGG16(TGModel):
    def __init__(self, x_train, y_train, input_type='poisson'):
        super(VGG16, self).__init__()

        # Check input size
        if x_train.shape[1] < 32 or x_train.shape[2] < 32:
            raise ValueError('input must be at least 32x32')

        # Define TensorFlow model
        tf_model = models.Sequential([
            layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, input_shape=x_train.shape[1:]),
            layers.Dropout(0.3),
            layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False),
            layers.AveragePooling2D(2),

            layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=False),
            layers.Dropout(0.4),
            layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=False),
            layers.AveragePooling2D(2),

            layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False),
            layers.Dropout(0.4),
            layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False),
            layers.Dropout(0.4),
            layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False),
            layers.AveragePooling2D(2),

            layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False),
            layers.Dropout(0.4),
            layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False),
            layers.Dropout(0.4),
            layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False),
            layers.AveragePooling2D(2),

            layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False),
            layers.Dropout(0.4),
            layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False),
            layers.Dropout(0.4),
            layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False),
            layers.AveragePooling2D(2),

            layers.Flatten(),
            layers.Dense(4096, activation="relu", use_bias=False),
            layers.Dropout(0.5),
            layers.Dense(4096, activation="relu", use_bias=False),
            layers.Dropout(0.5),
            layers.Dense(y_train.max() + 1, activation="softmax", use_bias=False),
        ], name='vgg16')

        # Train and convert model
        #tf_model = models.load_model('vgg16_tf_model')
        tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        tf_model.fit(x_train, y_train, batch_size=256, epochs=200)
        #models.save_model(tf_model, 'vgg16_tf_model', save_format='h5')
        self.convert_tf_model(tf_model, input_type)
        self.tf_model = tf_model


if __name__ == '__main__':
    args = parse_arguments('VGG16 classifier model')
    print('arguments: ' + str(vars(args)))

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Retrieve and normalise CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train[:args.n_train_samples] / 255.0
    y_train = y_train[:args.n_train_samples, 0]
    x_test = x_test[:args.n_test_samples] / 255.0
    y_test = y_test[:args.n_test_samples, 0]
    x_norm = x_train[np.random.choice(x_train.shape[0], args.n_norm_samples, replace=False)]

    # Create, normalise and evaluate TensorGeNN model
    tg_model = VGG16(x_train, y_train, input_type=args.input_type)
    tg_model.tf_model.evaluate(x_test, y_test)
    tg_model.compile(dt=args.dt, rng_seed=args.rng_seed,
                     batch_size=args.batch_size, share_weights=args.share_weights)
    if args.norm_method == 'data-norm':
        norm = DataNorm(x_norm, tg_model.tf_model)
        norm.normalize(tg_model)
    elif args.norm_method == 'spike-norm':
        norm = SpikeNorm(x_norm)
        norm.normalize(tg_model, args.classify_time)
    acc, spk_i, spk_t = tg_model.evaluate([x_test], [y_test], args.classify_time,
                                          save_samples=args.save_samples)

    # Report TensorGeNN model results
    print('Accuracy of VGG16 GeNN model: {}%'.format(acc[0]))
    if args.plot:
        names = ['input_nrn'] + [name + '_nrn' for name in tg_model.layer_names]
        neurons = [tg_model.g_model.neuron_populations[name] for name in names]
        raster_plot(spk_i, spk_t, neurons)
