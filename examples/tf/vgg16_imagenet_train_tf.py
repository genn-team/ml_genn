import os
import glob
from time import perf_counter
import tensorflow as tf
from tensorflow_addons.optimizers import SGDW

from arguments import parse_arguments
from imagenet_dataset import *

data_path = os.path.expanduser('/mnt/data0/imagenet')
train_path = os.path.join(data_path, 'train')
validate_path = os.path.join(data_path, 'validation')
checkpoint_path = './vgg16_imagenet_checkpoints'

batch_size = 256
epochs = 120
momentum = 0.9

dropout_rate = 0.25

def init(shape, dtype=None):
    stddev = tf.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)

def vgg16_block(x, n, filters, downsample=2, dropout_rate=dropout_rate):
    for i in range(n):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False,
                                   kernel_initializer=init)(x)
        x = tf.keras.layers.ReLU()(x)
        if i < (n - 1):
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        else:
            x = tf.keras.layers.AveragePooling2D(downsample)(x)
    return x

def vgg16():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))

    x = vgg16_block(inputs, 2, 64, downsample=2, dropout_rate=dropout_rate)
    x = vgg16_block(x, 2, 128, downsample=2, dropout_rate=dropout_rate)
    x = vgg16_block(x, 3, 256, downsample=2, dropout_rate=dropout_rate)
    x = vgg16_block(x, 3, 512, downsample=2, dropout_rate=dropout_rate)
    x = vgg16_block(x, 3, 512, downsample=2, dropout_rate=dropout_rate)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, use_bias=False)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(4096, use_bias=False)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(1000, use_bias=False)(x)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='vgg16_imagenet')

    return model


if __name__ == '__main__':
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    args = parse_arguments('VGG16 classifier')
    print('arguments: ' + str(vars(args)))

    # training dataset
    train_ds = imagenet_dataset_train(train_path)
    tf_train_ds = train_ds.batch(batch_size)
    tf_train_ds = tf_train_ds.prefetch(tf.data.AUTOTUNE)

    # validation dataset
    validate_ds = imagenet_dataset_validate(validate_path)
    tf_validate_ds = validate_ds.batch(batch_size)
    tf_validate_ds = tf_validate_ds.prefetch(tf.data.AUTOTUNE)

    # If there are any existing checkpoints
    existing_checkpoints = list(sorted(glob.glob(os.path.join(checkpoint_path, '*.h5'))))
    if len(existing_checkpoints) > 0:

        # Get file containing newest checkpoint
        newest_checkpoint_file = existing_checkpoints[-1]

        # Extract epoch number from checkpoint
        existing_checkpoint_title = os.path.splitext(os.path.basename(newest_checkpoint_file))[0]
        initial_epoch = int(existing_checkpoint_title.split('.')[0])
    else:
        newest_checkpoint_file = None
        initial_epoch = 0

    steps_per_epoch = 1281167 // epochs
    if initial_epoch < 30:
        steps = [(30 - initial_epoch) * steps_per_epoch, 
                 (60 - initial_epoch) * steps_per_epoch,
                 (90 - initial_epoch) * steps_per_epoch]
        decay = [1.0, 0.1, 0.01, 0.001]
        lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            steps, [0.05 * d for d in decay])
        wd_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            steps, [0.0001 * d for d in decay])
    elif initial_epoch < 60:
        steps = [(60 - initial_epoch) * steps_per_epoch,
                 (90 - initial_epoch) * steps_per_epoch]
        decay = [0.1, 0.01, 0.001]
        lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            steps, [0.05 * d for d in decay])
        wd_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            steps, [0.0001 * d for d in decay])
    elif initial_epoch < 90:
        steps = [(90 - initial_epoch) * steps_per_epoch]
        decay = [0.01, 0.001]
        lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            steps, [0.05 * d for d in decay])
        wd_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            steps, [0.0001 * d for d in decay])
    else:
        lr_schedule = 0.05 * 0.001
        wd_schedule = 0.0001 * 0.001

    # Create and compile TF model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        tf_model = vgg16()
        optimizer = SGDW(learning_rate=lr_schedule, momentum=momentum, nesterov=True, weight_decay=wd_schedule)
        tf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if args.reuse_tf_model:
        # Load old weights
        tf_model.load_weights('vgg16_imagenet_tf_weights.h5')

    else:
        # Load newest checkpoint weights if present
        if newest_checkpoint_file is not None:
            print(f'Loading epoch {initial_epoch} from checkpoint {newest_checkpoint_file}')
            tf_model.load_weights(newest_checkpoint_file)

        # Fit TF model
        callbacks = [tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, '{epoch:03d}.h5'), save_weights_only=True)]
        if args.record_tensorboard:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1))
        tf_model.fit(tf_train_ds, validation_data=tf_validate_ds, epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks)

        # Save weights
        tf_model.save_weights('vgg16_imagenet_tf_weights.h5')

    # Evaluate TF model
    tf_eval_start_time = perf_counter()
    tf_model.evaluate(tf_validate_ds)
    print("TF evaluation time: %f" % (perf_counter() - tf_eval_start_time))
