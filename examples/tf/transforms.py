# The following preprocessing functions are adapted from code freely available at:
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/transforms.lua

import tensorflow as tf


def color_normalize(mean, std):
    # samplewise norm (scalars) or colorwise norm (3-vectors)
    # input (height, width, channels) or (batch, height, width, channels)

    def fn(image, label):
        image = tf.math.subtract(image, mean)
        image = tf.math.divide(image, std)
        return image, label

    return fn


def horizontal_flip():
    # Flip image left to right with 0.5 probability
    # input (height, width, channels) or (batch, height, width, channels)

    def fn(image, label):
        image = tf.image.random_flip_left_right(image)
        return image, label

    return fn


def scale_small_edge(size):
    # Scales the image such that the smaller edge is equal to size

    def fn(image, label):
        image_shape = tf.shape(image)
        if image_shape[0] < image_shape[1]:
            new_shape = (size, image_shape[1]/image_shape[0] * size)
            image = tf.image.resize(image, new_shape, method='bicubic')
        else:
            new_shape = (image_shape[0]/image_shape[1] * size, size)
            image = tf.image.resize(image, new_shape, method='bicubic')
        return image, label

    return fn


def center_crop(size):
    # Crop to centered rectangle

    def fn(image, label):
        image_shape = tf.shape(image)
        h_offset = tf.cast(tf.math.ceil((image_shape[0] - size) / 2), 'int32')
        w_offset = tf.cast(tf.math.ceil((image_shape[1] - size) / 2), 'int32')
        image = tf.image.crop_to_bounding_box(image, h_offset, w_offset, size, size)
        return image, label

    return fn


def random_crop(size, padding=0):
    # Random crop from larger image with optional zero padding

    def fn(image, label):
        image_shape = tf.shape(image)
        pad_h = image_shape[0] + 2 * padding
        pad_w = image_shape[1] + 2 * padding
        image = tf.image.resize_with_crop_or_pad(image, pad_h, pad_w)
        image = tf.image.random_crop(image, (size, size, 3))
        return image, label

    return fn


# Random crop with size 8% to 100% and aspect ratio 3/4 to 4/3 (Inception-style)
def random_sized_crop(size):
    scale_small_edge_fn = scale_small_edge(size)
    center_crop_fn = center_crop(size)

    def fn(image, label):
        image_shape = tf.shape(image)
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            image_shape, tf.zeros((0, 0, 4)),
            min_object_covered=0.0,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.08, 1.0],
            max_attempts=10,
            use_image_if_no_bounding_boxes=True)
        is_bad = tf.reduce_all(tf.equal(image_shape[0:2], bbox_size[0:2]))

        if is_bad:
            # Bad bbox: scale small edge and center crop
            image, label = scale_small_edge_fn(image, label)
            image, label = center_crop_fn(image, label)
            #tf.print('random_sized_crop: BAD bbox')
            return image, label

        else:
            # Good bbox: crop image with bbox and resize
            image = tf.slice(image, bbox_begin, bbox_size)
            image = tf.image.resize(image, (size, size), method='bicubic')
            #tf.print('random_sized_crop: GOOD bbox:', bbox_begin, bbox_size)
            return image, label

    return fn

# Fixed order colour jitter
def colour_jitter(clamp):
    def fn(image, label):
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)

        if clamp:
            image - tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    return fn

# Lighting noise (AlexNet-style PCA-based noise)
def lighting(alphastd, eigval, eigvec):

    def fn(image, label):
        alpha = tf.random.normal((3,), stddev=alphastd)
        alpha_eigval = tf.reshape(alpha * eigval, (3, 1))
        rgb = tf.reshape(tf.matmul(eigvec, alpha_eigval), (3,))
        image += rgb
        #tf.print('lighting: RGB vector:', rgb)
        return image, label

    return fn
