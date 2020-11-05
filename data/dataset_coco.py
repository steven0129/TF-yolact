"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os

import tensorflow as tf

from data import anchor
from data import yolact_parser


# Todo encapsulate it as a class, here is the place to get dataset(train, eval, test)
def prepare_dataloader(tfrecord_dir, batch_size, img_size, subset="train"):

    anchorobj = anchor.Anchor(img_size=img_size,
                              feature_map_size=[32, 16, 8, 4, 2],
                              aspect_ratio=[1, 0.5, 2],
                              scale=[24, 48, 96, 192, 384])

    parser = yolact_parser.Parser(output_size=img_size,
                                  anchor_instance=anchorobj,
                                  match_threshold=0.7,
                                  unmatched_threshold=0.3,
                                  proto_output_size=64,
                                  mode=subset)
    files = tf.io.matching_files(os.path.join(tfrecord_dir, "obj_%s.*" % subset))
    num_shards = tf.cast(tf.shape(files)[0], tf.int64)
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(num_shards)
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset,
                                cycle_length=num_shards,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=2048)
    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def prepare_evalloader(tfrecord_dir, img_size, subset="train"):

    anchorobj = anchor.Anchor(img_size=img_size,
                              feature_map_size=[32, 16, 8, 4, 2],
                              aspect_ratio=[1, 0.5, 2],
                              scale=[24, 48, 96, 192, 384])

    parser = yolact_parser.Parser(output_size=img_size,
                                  anchor_instance=anchorobj,
                                  match_threshold=0.5,
                                  unmatched_threshold=0.5,
                                  proto_output_size=64,
                                  mode=subset)
    files = tf.io.matching_files(os.path.join(tfrecord_dir, "obj_%s.*" % subset))
    num_shards = tf.cast(tf.shape(files)[0], tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices(files)

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=num_shards,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
