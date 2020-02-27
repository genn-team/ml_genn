import tensorflow as tf
import tensor_genn as tg
import argparse

class SimpleMNIST(tg.TGModel):
    def __init__(self, dt=1.0, input_type='poisson', rate_factor=1.0, rng_seed=0):
        super(SimpleMNIST, self).__init__()

        # Define TensorFlow model
        tf_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, 5, padding='valid', activation='relu', use_bias=False, input_shape=(28, 28, 1)),
            tf.keras.layers.AveragePooling2D(2),
            tf.keras.layers.Conv2D(8, 5, padding='same', activation='relu', use_bias=False),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', use_bias=False),
            tf.keras.layers.Dense(64, activation='relu', use_bias=False),
            tf.keras.layers.Dense(10, activation='softmax', use_bias=False)
        ], name='simple_mnist')

        # Retrieve and normalise dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = self.x_train.reshape((-1, 28, 28, 1)) / 255.0
        self.x_test = self.x_test.reshape((-1, 28, 28, 1)) / 255.0

        # Train and convert model
        tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        tf_model.fit(self.x_train, self.y_train, epochs=5)
        tf_model.evaluate(self.x_test, self.y_test)
        self.convert_tf_model(tf_model, dt=dt, input_type=input_type, rate_factor=rate_factor, rng_seed=rng_seed)

    def evaluate(self, **kwargs):
        return super(SimpleMNIST, self).evaluate(self.x_test, self.y_test, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple MNIST classifier model')
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--input-type', type=str, default='poisson', choices=tg.supported_input_types)
    parser.add_argument('--rate-factor', type=float, default=1.0)
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--classify-time', type=float, default=500.0)
    parser.add_argument('--classify-spikes', type=int, default=100)
    parser.add_argument('--save-samples', type=int, default=[], action='append')
    parser.add_argument('--model-norm-samples', type=int, default=256)
    args = parser.parse_args()

    model = SimpleMNIST(dt=args.dt, input_type=args.input_type,
                        rate_factor=args.rate_factor, rng_seed=args.rng_seed)
    accuracy, spike_ids, spike_times = model.evaluate(classify_time=args.classify_time,
                                                      classify_spikes=args.classify_spikes,
                                                      save_samples=args.save_samples)
    print('Accuracy of simple MNIST GeNN model: {}%'.format(accuracy))
