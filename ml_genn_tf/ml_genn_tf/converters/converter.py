import tensorflow as tf

class Converter:
    def validate_tf_layer(self, tf_layer, config):
        if isinstance(tf_layer, (tf.keras.layers.Dense,
                                 tf.keras.layers.Conv2D)):
            if tf_layer.use_bias:
                # no bias tensors allowed
                raise NotImplementedError("Bias tensors not supported")

            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError("Output layer must have ReLU "
                                              "or softmax activation")

            elif config.has_activation:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError("Hidden layers must have "
                                              "ReLU activation")

        elif isinstance(tf_layer, tf.keras.layers.Activation):
            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError("Output layer must have "
                                              "ReLU or softmax activation")

            else:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError("Hidden layers must have "
                                              "ReLU activation")

        elif isinstance(tf_layer, tf.keras.layers.ReLU):
            # ReLU activation allowed everywhere
            pass

        elif isinstance(tf_layer, tf.keras.layers.Softmax):
            # softmax activation only allowed for output layers
            if not config.is_output:
                raise NotImplementedError("Only output layers "
                                          "may use softmax")

        elif isinstance(tf_layer, tf.keras.layers.GlobalAveragePooling2D):
            # global average pooling allowed
            pass
        elif isinstance(tf_layer, tf.keras.layers.AveragePooling2D):
            if tf_layer.padding != 'valid':
                raise NotImplementedError("Only valid padding is supported "
                                          "for pooling layers")
        else:
            # no other layers allowed
            raise NotImplementedError(f"{tf_layer.__class__.__name__} "
                                      f"layers are not supported")