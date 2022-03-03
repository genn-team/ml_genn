import numpy as np
from transforms import *

cifar10_mean = np.array([125.3, 123.0, 113.9], dtype='float32')
cifar10_std = np.array([63.0, 62.1, 66.7], dtype='float32')

def cifar10_dataset_train():
    (train_x, train_y), _ = tf.keras.datasets.cifar10.load_data()
    train_x = train_x.astype('float32')

    color_normalize_fn = color_normalize(cifar10_mean, cifar10_std)
    random_crop_fn = random_crop(32, 4)
    horizontal_flip_fn = horizontal_flip()

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(random_crop_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(horizontal_flip_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds

def cifar10_dataset_validate():
    _, (validate_x, validate_y) = tf.keras.datasets.cifar10.load_data()
    validate_x = validate_x.astype('float32')

    color_normalize_fn = color_normalize(cifar10_mean, cifar10_std)

    validate_ds = tf.data.Dataset.from_tensor_slices((validate_x, validate_y))
    validate_ds = validate_ds.map(color_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return validate_ds
