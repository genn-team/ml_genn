import numpy as np
import tensorflow as tf
from data_norm import DataNorm
from spike_norm import SpikeNorm
from tensor_genn.utils.plotting import raster_plot

from model import TGModel

def train_mnist(x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 5, padding='valid', strides=1, activation='relu', use_bias=False, input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, 5, padding='valid', strides=1, activation='relu', use_bias=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', use_bias=False),
        tf.keras.layers.Dense(64, activation='relu', use_bias=False),
        tf.keras.layers.Dense(10, activation='softmax', use_bias=False)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1)
    model.evaluate(x_test, y_test)

    return model


def main():
    tf.get_logger().setLevel('INFO')

    # Retrieve and normalise dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape((-1, 28, 28, 1)), x_test.reshape((-1, 28, 28, 1))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    #n_train = 10000
    #x_train = x_train[:n_train]
    #y_train = y_train[:n_train]

    #x_norm = x_train
    x_norm = x_train[:256]

    n_test = 1000
    x_test = x_test[:n_test]
    y_test = y_test[:n_test]

    pres_t = 50.0
    #pres_t = 100.0
    #pres_t = 150.0
    #pres_t = 200.0
    #pres_t = 250.0
    #pres_t = 300.0
    #pres_t = 1000.0
    #pres_t = 2500.0

    # Create / save / load TF model
    model_name = './mnist.h5'
    #tf_model = train_mnist(x_train, y_train, x_test, y_test)
    #tf.keras.models.save_model(tf_model, model_name)
    tf_model = tf.keras.models.load_model(model_name)
    tf_model.evaluate(x_test, y_test)
    #print(tf_model.summary())
    for layer in tf_model.layers:
        print(layer)
        print('in shape:  ' + str(layer.input_shape))
        print('out shape: ' + str(layer.output_shape))
        if len(layer.get_weights()) > 0: print('w shape:   ' + str(layer.get_weights()[0].shape))
    #return

    tg_model = TGModel(tf_model)
    tg_model.create_genn_model(dt=1.0)

    # accuracy, spike_ids, spike_times = tg_model.evaluate_genn_model(x_test, y_test, present_time=pres_t, save_samples=[0])
    # print('Accuracy achieved by GeNN model: {}%'.format(accuracy))

    # # ===== SPIKE NORM =====
    # norm = SpikeNorm(x_norm, present_time=pres_t)
    # norm.normalize(tg_model)

    # ===== DATA NORM =====
    norm = DataNorm(x_norm, batch_size=100)
    norm.normalize(tg_model)

    accuracy, spike_ids, spike_times = tg_model.evaluate_genn_model(x_test, y_test, present_time=pres_t, save_samples=[0])
    print('Accuracy achieved by GeNN model: {}%'.format(accuracy))

    names = ['input_nrn'] + [name + '_nrn' for name in tg_model.layer_names]
    neurons = [tg_model.g_model.neuron_populations[name] for name in names]
    raster_plot(spike_ids, spike_times, neurons)


if __name__ == '__main__':
    main()
