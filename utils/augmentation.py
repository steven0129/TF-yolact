import numpy as np
import tensorflow as tf

from utils import utils

"""
Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/preprocessing/ssd_vgg_preprocessing.py
"""

def photometric_distortion(image):
    color_ordering = tf.random.uniform([1], minval=0, maxval=4)[0]

    if color_ordering < 1 and color_ordering > 0:
        image = tf.image.random_brightness(image, max_delta=0.12)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    elif color_ordering < 2 and color_ordering > 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.125)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    elif color_ordering < 3 and color_ordering > 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=0.125)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    else:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.125)

    return image


def horizontal_flip(image, bboxes, masks):
    # Random mirroring (img, bbox, mask)
    image = tf.image.flip_left_right(image)
    masks = tf.image.flip_left_right(masks)
    bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                       bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
    return image, bboxes, masks


def random_augmentation(img, bboxes, masks, output_size, proto_output_size, classes):
    # generate random
    FLAGS = tf.random.uniform([2], minval=0, maxval=1)
    FLAG_PHOTO_DISTORTION = FLAGS[0]
    FLAG_HOR_FLIP = FLAGS[1]
    
    # Random Photometric Distortions (img)
    if FLAG_PHOTO_DISTORTION > 0.5:
        img = photometric_distortion(img)

    if FLAG_HOR_FLIP > 0.5:
        img, bboxes, masks = horizontal_flip(img, bboxes, masks)

    # resize masks to protosize
    masks = tf.image.resize(masks, [proto_output_size, proto_output_size], method=tf.image.ResizeMethod.BILINEAR)
    masks = tf.cast(masks + 0.5, tf.int64)
    masks = tf.squeeze(masks)
    masks = tf.cast(masks, tf.float32)
    return img, bboxes, masks, classes
