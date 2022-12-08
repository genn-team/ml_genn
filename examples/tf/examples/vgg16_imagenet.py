import os
from six import iteritems
from time import perf_counter
import tensorflow as tf

from ml_genn import Model
from ml_genn.utils import parse_arguments, raster_plot
from vgg16_imagenet_train_tf import vgg16
from imagenet_dataset import *

data_path = os.path.expanduser('/mnt/data0/imagenet')
train_path = os.path.join(data_path, 'train')
validate_path = os.path.join(data_path, 'validation')
checkpoint_path = './vgg16_imagenet_checkpoints'


if __name__ == '__main__':
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    args = parse_arguments('VGG16 classifier')
    print('arguments: ' + str(vars(args)))

    # ML GeNN norm dataset
    mlg_norm_ds = imagenet_dataset_train(train_path)
    mlg_norm_ds = mlg_norm_ds.take(args.n_norm_samples)
    mlg_norm_ds = mlg_norm_ds.batch(args.batch_size)
    mlg_norm_ds = mlg_norm_ds.prefetch(tf.data.AUTOTUNE)

    # ML GeNN validation dataset
    if args.n_test_samples is None:
        args.n_test_samples = 50000
    mlg_validate_ds = imagenet_dataset_validate(validate_path)
    mlg_validate_ds = mlg_validate_ds.take(args.n_test_samples)
    mlg_validate_ds = mlg_validate_ds.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
    mlg_validate_ds = mlg_validate_ds.batch(args.batch_size)
    mlg_validate_ds = mlg_validate_ds.prefetch(tf.data.AUTOTUNE)

    # Create and compile TF model
    with tf.device('/CPU:0'):
        tf_model = vgg16()
        tf_model.compile()
        tf_model.load_weights('vgg16_imagenet_tf_weights.h5')

    # Create a suitable converter to convert TF model to ML GeNN
    K = 10
    T = 2500
    converter = args.build_converter(mlg_norm_ds, signed_input=True, K=10, norm_time=2500)

    # Convert and compile ML GeNN model
    mlg_model = Model.convert_tf_model(
        tf_model, converter=converter, connectivity_type=args.connectivity_type,
        dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed,
        kernel_profiling=args.kernel_profiling)

    # Evaluate ML GeNN model
    time = K if args.converter == 'few-spike' else T
    mlg_eval_start_time = perf_counter()
    acc, spk_i, spk_t = mlg_model.evaluate_batched(
        mlg_validate_ds, time, save_samples=args.save_samples)
    print("MLG evaluation time: %f" % (perf_counter() - mlg_eval_start_time))

    if args.kernel_profiling:
        print("Kernel profiling:")
        for n, t in iteritems(mlg_model.get_kernel_times()):
            print("\t%s: %fs" % (n, t))

    # Report ML GeNN model results
    print(f'Accuracy of VGG16 GeNN model: {acc[0]}%')
    if args.plot:
        neurons = [l.neurons.nrn for l in mlg_model.layers]
        raster_plot(spk_i, spk_t, neurons, time=time)
