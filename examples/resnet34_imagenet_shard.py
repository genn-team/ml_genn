import os
import glob
from six import iteritems
from time import perf_counter
import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import SGDW

from ml_genn import Model
from ml_genn.converters import DataNorm, SpikeNorm, FewSpike, Simple
from ml_genn.utils import parse_arguments, raster_plot
from transforms import *


data_path = os.path.expanduser('~/../shared/data/imagenet')
#data_path = os.path.expanduser('/mnt/data0/imagenet')

train_path = os.path.join(data_path, 'train')
validate_path = os.path.join(data_path, 'validation')
checkpoint_path = './resnet34_imagenet_checkpoints'

batch_size = 256
epochs = 120
momentum = 0.9


def init_non_residual(shape, dtype=None):
    stddev = tf.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)


def init_residual(shape, dtype=None):
    stddev = tf.sqrt(2.0) / float(shape[0] * shape[1] * shape[3])
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)


def resnet34_block(input_layer, filters, downsample=False):
    stride = 2 if downsample else 1

    x = tf.keras.layers.Conv2D(filters, 3, padding='same', strides=stride, use_bias=False,
                               kernel_initializer=init_residual)(input_layer)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', strides=1, use_bias=False,
                               kernel_initializer=init_residual)(x)

    if downsample:
        res = tf.keras.layers.Conv2D(filters, 1, padding='same', strides=2, use_bias=False)(input_layer)
    else:
        res = input_layer

    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.ReLU()(x)

    return x


def resnet34():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))

    x = tf.keras.layers.Conv2D(64, 3, padding='same', strides=1, use_bias=False,
                               kernel_initializer=init_non_residual)(inputs)
    x = tf.keras.layers.ReLU()(x)

    # Sengupta et al (2019) use two extra plain pre-processing layers,
    # so this technically isn't ResNet34.
    x = tf.keras.layers.Conv2D(64, 3, padding='same', strides=1, use_bias=False,
                               kernel_initializer=init_non_residual)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', strides=2, use_bias=False,
                               kernel_initializer=init_non_residual)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.AveragePooling2D(2)(x)

    x = resnet34_block(x, 64)
    x = resnet34_block(x, 64)
    x = resnet34_block(x, 64)

    x = resnet34_block(x, 128, True)
    x = resnet34_block(x, 128)
    x = resnet34_block(x, 128)
    x = resnet34_block(x, 128)

    x = resnet34_block(x, 256, True)
    x = resnet34_block(x, 256)
    x = resnet34_block(x, 256)
    x = resnet34_block(x, 256)
    x = resnet34_block(x, 256)
    x = resnet34_block(x, 256)

    x = resnet34_block(x, 512, True)
    x = resnet34_block(x, 512)
    x = resnet34_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1000, use_bias=False)(x)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='resnet34_imagenet')

    return model


def parse_buffer(buffer):
    keys_to_features = {
        "image/encoded": tf.io.FixedLenFeature((), tf.string, ''),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64, -1)}
    parsed = tf.io.parse_single_example(buffer, keys_to_features)

    # get label
    label = tf.cast(tf.reshape(parsed["image/class/label"], shape=[]), dtype=tf.int32) - 1

    # decode image
    image = tf.image.decode_jpeg(tf.reshape(parsed["image/encoded"], shape=[]), channels=3)

    # convert to float **NOTE** this converts [0,255] to [0,1) range
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label


for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

#if __name__ == '__main__':
with tf.device('/CPU:0'):
    args = parse_arguments('ResNet34 classifier')
    print('arguments: ' + str(vars(args)))

    print(f'###################### SHARD {args.shard} ######################')

    # load ImageNet 2012 data
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype='float32')
    imagenet_eigval = np.array([0.2175, 0.0188, 0.0045], dtype='float32')
    imagenet_eigvec = np.array([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ], dtype='float32')

    # training dataset
    random_sized_crop_fn = random_sized_crop(224)
    lighting_fn = lighting(0.1, imagenet_eigval, imagenet_eigvec)
    color_normalize_fn = color_normalize(imagenet_mean, imagenet_std)
    horizontal_flip_fn = horizontal_flip()

    train_ds = tf.data.Dataset.list_files(os.path.join(train_path, 'train*'))
    train_ds = train_ds.shuffle(len(train_ds))
    train_ds = train_ds.interleave(tf.data.TFRecordDataset, cycle_length=8, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=8192)
    train_ds = train_ds.map(parse_buffer, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(random_sized_crop_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lighting_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(horizontal_flip_fn, num_parallel_calls=tf.data.AUTOTUNE)
    tf_train_ds = train_ds.batch(batch_size)
    tf_train_ds = tf_train_ds.prefetch(tf.data.AUTOTUNE)

    # validation dataset
    scale_small_edge_fn = scale_small_edge(256)
    center_crop_fn = center_crop(224)
    color_normalize_fn = color_normalize(imagenet_mean, imagenet_std)

    validate_ds = tf.data.Dataset.list_files(os.path.join(validate_path, 'validation*'), shuffle=False)
    validate_ds = validate_ds.interleave(tf.data.TFRecordDataset, cycle_length=8, num_parallel_calls=tf.data.AUTOTUNE)
    validate_ds = validate_ds.map(parse_buffer, num_parallel_calls=tf.data.AUTOTUNE)
    validate_ds = validate_ds.map(scale_small_edge_fn, num_parallel_calls=tf.data.AUTOTUNE)
    validate_ds = validate_ds.map(center_crop_fn, num_parallel_calls=tf.data.AUTOTUNE)
    validate_ds = validate_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    tf_validate_ds = validate_ds.batch(batch_size)
    tf_validate_ds = tf_validate_ds.prefetch(tf.data.AUTOTUNE)

    # ML GeNN norm dataset
    mlg_norm_ds = tf.data.Dataset.list_files(os.path.join(train_path, 'train*'), shuffle=False)
    mlg_norm_ds = mlg_norm_ds.shuffle(buffer_size=len(mlg_norm_ds), seed=122) # fix seed, so norm set is same accross nodes
    mlg_norm_ds = mlg_norm_ds.interleave(tf.data.TFRecordDataset)#, cycle_length=8, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_norm_ds = mlg_norm_ds.shuffle(buffer_size=8192, seed=822) # fix seed, so norm set is same accross nodes
    mlg_norm_ds = mlg_norm_ds.map(parse_buffer, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_norm_ds = mlg_norm_ds.map(scale_small_edge_fn, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_norm_ds = mlg_norm_ds.map(center_crop_fn, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_norm_ds = mlg_norm_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_norm_ds = mlg_norm_ds.take(args.n_norm_samples).as_numpy_iterator()
    norm_x = np.array([d[0] for d in mlg_norm_ds])

    # ML GeNN validation dataset
    if args.n_test_samples is None:
        #args.n_test_samples = 50000
        args.n_test_samples = 6250 # = 50000 / 8, since we use 8 jade2 nodes
    mlg_validate_ds = tf.data.Dataset.list_files(os.path.join(validate_path, 'validation*'), shuffle=False)
    mlg_validate_ds = mlg_validate_ds.interleave(tf.data.TFRecordDataset)#, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_validate_ds = mlg_validate_ds.shard(8, args.shard) # shard val ds for 8 nodes
    mlg_validate_ds = mlg_validate_ds.map(parse_buffer, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_validate_ds = mlg_validate_ds.map(scale_small_edge_fn, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_validate_ds = mlg_validate_ds.map(center_crop_fn, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_validate_ds = mlg_validate_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    mlg_validate_ds = mlg_validate_ds.take(args.n_test_samples)
    mlg_validate_ds = mlg_validate_ds.batch(args.batch_size)
    mlg_validate_ds = mlg_validate_ds.prefetch(tf.data.AUTOTUNE)
    mlg_validate_ds = mlg_validate_ds.as_numpy_iterator()
    
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

    steps_per_epoch = 1281167 // batch_size
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
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    if True:
        tf_model = resnet34()
        optimizer = SGDW(learning_rate=lr_schedule, momentum=0.9, nesterov=True, weight_decay=wd_schedule)
        tf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if args.reuse_tf_model:
        # Load old weights
        tf_model.load_weights('../../resnet34_imagenet_tf_weights.h5')

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
        tf_model.save_weights('resnet34_imagenet_tf_weights.h5')

    # # Evaluate TF model
    # tf_eval_start_time = perf_counter()
    # tf_model.evaluate(tf_validate_ds)
    # print("TF evaluation time: %f" % (perf_counter() - tf_eval_start_time))

    # Create a suitable converter to convert TF model to ML GeNN
    converter = args.build_converter(norm_x, signed_input=True, K=10, norm_time=2500)

    # Convert and compile ML GeNN model
    mlg_model = Model.convert_tf_model(
        tf_model, converter=converter, connectivity_type=args.connectivity_type,
        dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed,
        kernel_profiling=args.kernel_profiling)

    # Evaluate ML GeNN model
    time = 10 if args.converter == 'few-spike' else 2500
    mlg_eval_start_time = perf_counter()
    #acc, spk_i, spk_t = mlg_model.evaluate([validate_x], [validate_y], time, save_samples=args.save_samples)
    acc, spk_i, spk_t = mlg_model.evaluate_iterator(mlg_validate_ds, args.n_test_samples, time, save_samples=args.save_samples)
    print("MLG evaluation time: %f" % (perf_counter() - mlg_eval_start_time))

    if args.kernel_profiling:
        print("Kernel profiling:")
        for n, t in iteritems(mlg_model.get_kernel_times()):
            print("\t%s: %fs" % (n, t))

    # Report ML GeNN model results
    print(f'Accuracy of ResNet34 GeNN model: {acc[0]:.16f}%')
    if args.plot:
        neurons = [l.neurons.nrn for l in mlg_model.layers]
        raster_plot(spk_i, spk_t, neurons, time=time)
