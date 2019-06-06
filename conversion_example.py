import tensorflow as tf

import tensor_genn as tg
from tensor_genn.algorithms import ReLUANN

def train_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu', use_bias=False),
    tf.keras.layers.Dense(64, activation='relu', use_bias=False),
    tf.keras.layers.Dense(10, activation='softmax',use_bias=False)
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test[:1000], y_test[:1000])

    return model, x_train*255., y_train, x_test*255., y_test

tf_model, x_train, y_train, x_test, y_test = train_mnist()

# Create models
relu_ann = ReLUANN()
g_model = tg.convert_model(tf_model,relu_ann,x_test[:1000],y_test[:1000])