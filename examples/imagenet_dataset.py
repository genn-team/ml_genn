import os
import numpy as np
import tensorflow as tf
from transforms import *

imagenet_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
imagenet_std = np.array([0.229, 0.224, 0.225], dtype='float32')
imagenet_eigval = np.array([0.2175, 0.0188, 0.0045], dtype='float32')
imagenet_eigvec = np.array([
    [-0.5675,  0.7192,  0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948,  0.4203],
], dtype='float32')

def parse_tfrecord(example):
    keys_to_features = {
        "image/encoded": tf.io.FixedLenFeature((), tf.string, ''),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64, -1)}
    parsed = tf.io.parse_single_example(example, keys_to_features)

    # get label
    label = tf.cast(tf.reshape(parsed["image/class/label"], shape=[]), dtype=tf.int32) - 1

    # decode image
    image = tf.image.decode_jpeg(tf.reshape(parsed["image/encoded"], shape=[]), channels=3)

    # convert to float **NOTE** this converts [0,255] to [0,1) range
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label

def imagenet_dataset_train(train_path):
    random_sized_crop_fn = random_sized_crop(224)
    lighting_fn = lighting(0.1, imagenet_eigval, imagenet_eigvec)
    color_normalize_fn = color_normalize(imagenet_mean, imagenet_std)
    horizontal_flip_fn = horizontal_flip()

    train_ds = tf.data.Dataset.list_files(os.path.join(train_path, 'train*'))
    train_ds = train_ds.shuffle(len(train_ds))
    train_ds = train_ds.interleave(tf.data.TFRecordDataset, cycle_length=8, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=8192)
    train_ds = train_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(random_sized_crop_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lighting_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(horizontal_flip_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds

def imagenet_dataset_validate(validate_path):
    scale_small_edge_fn = scale_small_edge(256)
    center_crop_fn = center_crop(224)
    color_normalize_fn = color_normalize(imagenet_mean, imagenet_std)

    validate_ds = tf.data.Dataset.list_files(os.path.join(validate_path, 'validation*'), shuffle=False)
    validate_ds = validate_ds.interleave(tf.data.TFRecordDataset, cycle_length=8, num_parallel_calls=tf.data.AUTOTUNE)
    validate_ds = validate_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    validate_ds = validate_ds.map(scale_small_edge_fn, num_parallel_calls=tf.data.AUTOTUNE)
    validate_ds = validate_ds.map(center_crop_fn, num_parallel_calls=tf.data.AUTOTUNE)
    validate_ds = validate_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return validate_ds
