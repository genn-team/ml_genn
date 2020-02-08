import numpy as np
import tensorflow as tf
import tensor_genn as tg
from tensor_genn.algorithms import ReLUANN
from tensor_genn.algorithms.weight_normalization import DataNorm
from tensor_genn.utils.plotting import raster_plot

def train_cifar10():
    cifar10 = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train_normed, x_test_normed = x_train / 255.0, x_test / 255.0
    x_train_normed, x_test_normed = x_train_normed.reshape((-1,32,32,3)), x_test_normed.reshape((-1,32,32,3))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,5,padding='valid',activation='relu',use_bias=False,input_shape=(32,32,3)),
        tf.keras.layers.AveragePooling2D(2),
        tf.keras.layers.Conv2D(8,5,padding='same',activation='relu',use_bias=False),
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(128, activation='relu', use_bias=False),
        tf.keras.layers.Dense(64, activation='relu', use_bias=False),
        tf.keras.layers.Dense(10, activation='softmax',use_bias=False)
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train_normed, y_train, epochs=0)

    model.evaluate(x_test_normed[:10000], y_test[:10000])

    print(model.summary())

    return model, x_train_normed, y_train, x_test_normed, y_test

def train_mnist(x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 5, padding='valid', strides=1, activation='relu', use_bias=False, input_shape=(28, 28, 1)),
        tf.keras.layers.AveragePooling2D(2),
        tf.keras.layers.Conv2D(8, 5, padding='same', strides=1, activation='relu', use_bias=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', use_bias=False),
        tf.keras.layers.Dense(64, activation='relu', use_bias=False),
        tf.keras.layers.Dense(10, activation='softmax', use_bias=False)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train[:10000], y_train[:10000], epochs=1)
    model.evaluate(x_test[:10000], y_test[:10000])

    return model


def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape((-1, 28, 28, 1)), x_test.reshape((-1, 28, 28, 1))

    tf_model = train_mnist(x_train, y_train, x_test, y_test)
    #tf.keras.models.save_model(tf_model, './mnist.h5')
    #tf_model = tf.keras.models.load_model('./mnist.h5')
    #print(tf_model.summary())

    for layer in tf_model.layers:
        print(layer)
        print('in shape:  ' + str(layer.input_shape))
        print('out shape: ' + str(layer.output_shape))
        if len(layer.get_weights()) > 0: print('w shape:   ' + str(layer.get_weights()[0].shape))


    # Sample conversion
    relu_ann = ReLUANN(single_example_time=100.0, dense_membrane_capacitance=0.1,
                       sparse_membrane_capacitance=0.5, neuron_threshold_voltage=-56.0)

    #relu_ann = ReLUANN(single_example_time=100.0, dense_membrane_capacitance=1.0,
    #                   sparse_membrane_capacitance=1.0, neuron_threshold_voltage=-56.0)

    data_norm = DataNorm(x_train, batch_size=100)

    g_model, spike_ids, spike_times, neuron_pops, syn_pops = tg.convert_model(
        tf_model, relu_ann, x_test[:100], y_test[:100], data_norm, [20,21]
        #tf_model, relu_ann, x_test[:10000], y_test[:10000], data_norm, [20,21]
    )

    raster_plot(spike_ids, spike_times, neuron_pops)


if __name__ == '__main__':
    main()
