import os
from six import iteritems
from time import perf_counter
import tensorflow as tf

from ml_genn.callbacks import SpikeRecorder

from arguments import parse_arguments
from plotting import plot_spikes
from vgg16_imagenet_train_tf import vgg16
from imagenet_dataset import *

data_path = os.path.expanduser('/usr/local/share/imagenet/shards/')
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
    converter = args.build_converter(mlg_norm_ds, signed_input=True, 
                                     k=K, evaluate_timesteps=T)

    # Convert and compile ML GeNN model
    net, net_inputs, net_outputs, tf_layer_pops = converter.convert(tf_model)

    # If we should plot any spikes, turn on spike recording for all populations
    if len(args.plot_sample_spikes) > 0:
        for l in tf_model.layers:
            if l in tf_layer_pops:
                callbacks.append(
                    SpikeRecorder(tf_layer_pops[l], key=l.name,
                                  example_filter=args.plot_sample_spikes))

    # Create suitable compiler for model
    compiler = converter.create_compiler(prefer_in_memory_connect=args.prefer_in_memory_connect,
                                         dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed, 
                                         kernel_profiling=args.kernel_profiling)
    compiled_net = compiler.compile(net, inputs=net_inputs, 
                                    outputs=net_outputs)
    compiled_net.genn_model.timing_enabled = args.kernel_profiling
    
    with compiled_net:
        # If we should plot any spikes, add spike recorder callback for all populations
        callbacks = ["batch_progress_bar"]
        if len(args.plot_sample_spikes) > 0:
            for p in net.populations:
                callbacks.append(SpikeRecorder(p, example_filter=args.plot_sample_spikes))
                
        # Evaluate ML GeNN model
        start_time = perf_counter()
        num_batches = args.n_test_samples // args.batch_size
        metrics, cb_data = compiled_net.evaluate_batch_iter(
            net_inputs, net_outputs, iter(mlg_validate_ds),
            num_batches=num_batches, callbacks=callbacks)
        end_time = perf_counter()
        print(f"Accuracy = {100.0 * metrics[net_outputs[0]].result}%")
        print(f"Time = {end_time - start_time} s")
        
        if args.kernel_profiling:
            reset_time = compiled_net.genn_model.get_custom_update_time("Reset")
            print(f"Kernel profiling:\n"
                  f"\tinit_time: {compiled_net.genn_model.init_time} s\n"
                  f"\tinit_sparse_time: {compiled_net.genn_model.init_sparse_time} s\n"
                  f"\tneuron_update_time: {compiled_net.genn_model.neuron_update_time} s\n"
                  f"\tpresynaptic_update_time: {compiled_net.genn_model.presynaptic_update_time} s\n"
                  f"\tcustom_update_reset_time: {reset_time} s")
        
        # Plot spikes if desired
        if len(args.plot_sample_spikes) > 0:
            plot_spikes(cb_data, args.plot_sample_spikes)
