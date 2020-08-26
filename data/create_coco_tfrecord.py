# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw COCO dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
import hashlib
import io
import json
import os
import cv2

import PIL.Image
import contextlib2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# use absl for tf 2.0
from absl import app
from absl import flags
from absl import logging
from pycocotools import mask
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from skimage.measure import regionprops

from data import dataset_util

FLAGS = flags.FLAGS

flags.DEFINE_boolean('include_masks', True,
                     'Whether to include instance segmentations masks (PNG encoded) in the result. default: False.')
flags.DEFINE_string('train_image_dir', 'train2017',
                    'Training image directory.')
flags.DEFINE_string('val_image_dir', 'val2017',
                    'Validation image directory.')
flags.DEFINE_string('test_image_dir', '',
                    'Test image directory.')
flags.DEFINE_string('train_annotations_file', 'annotations/instances_train2017.json',
                    'Training annotations JSON file.')
flags.DEFINE_string('val_annotations_file', 'annotations/instances_val2017.json',
                    'Validation annotations JSON file.')
flags.DEFINE_string('testdev_annotations_file', '',
                    'Test-dev annotations JSON file.')
flags.DEFINE_string('output_dir', './coco', 'Output data directory.')

logging.set_verbosity(logging.INFO)
dp_coco = COCO( f'data/DensePose/all.json' )

def GenDPMask(ann):
    mask_body = np.zeros([256, 256])
    mask_face = np.zeros([256, 256])
    for i in range(0, 13):
        if len(ann['dp_masks'][i]) != 0:
            curr_mask = mask_util.decode(ann['dp_masks'][i])
            mask_body[curr_mask > 0] = 1
    
    if len(ann['dp_masks'][13]) != 0:
        curr_mask = mask_util.decode(ann['dp_masks'][13])
        mask_face[curr_mask > 0] = 1

    return mask_face, mask_body

def create_tf_horizontal_flip_example(image,
                      annotations_list,
                      image_dir,
                      category_index):
    """Converts image and annotations to a tf.Example proto.

    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      annotations_list:
        list of dicts with keys:
        [u'segmentation', u'area', u'iscrowd', u'image_id',
        u'bbox', u'category_id', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
      image_dir: directory containing the image files.
      category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    r = 320
    image = image.resize((r, r), PIL.Image.ANTIALIAS)
    image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)   # Image Horizontal Flip
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='JPEG')
    encoded_jpg = bytes_io.getvalue()
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    dp_triggered = True  # DensePose preprocessing just one time

    for object_annotations in annotations_list:
        category_id = int(object_annotations['category_id'])

        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue

        if category_id == 1 and dp_triggered: # Person --> Person Face & Person Body
            try:
                img = dp_coco.loadImgs(image_id)[0]
                ann_ids = dp_coco.getAnnIds(imgIds=img['id'])
                anns = dp_coco.loadAnns(ann_ids)

                for ann in anns:
                    if 'dp_masks' in ann.keys():
                        mask_face, mask_body = GenDPMask(ann)
                        img_face = cv2.imread(f'{image_dir}/{filename}')
                        img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
                        img_body = img_face.copy()

                        bbr = np.array(ann['bbox']).astype(int)
                        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0]+bbr[2], bbr[1]+bbr[3]
                        x2 = min( [x2, img_face.shape[1]] )
                        y2 = min( [y2, img_face.shape[0]] )

                        # Person Face
                        if np.count_nonzero(mask_face) != 0:
                            category_id = -1 
                            category_ids.append(category_id)
                            category_names.append('Person Face'.encode('utf8'))

                            mask_face = cv2.resize(mask_face, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
                            mask_face = (mask_face == 1)
                            binary_mask = np.zeros((img_face.shape[0], img_face.shape[1]))
                            binary_mask[y1:y2, x1:x2][mask_face] = 1
                            binary_mask = np.uint8(binary_mask)
                            binary_mask = np.fliplr(binary_mask)   # Person Face Horizontal Flip
                            
                            props = regionprops(binary_mask)
                            prop = props[0]

                            xmin.append(float(prop.bbox[1]) / image_width)
                            xmax.append(float(prop.bbox[3]) / image_width)
                            ymin.append(float(prop.bbox[0]) / image_height)
                            ymax.append(float(prop.bbox[2]) / image_height)
                            is_crowd.append(ann['iscrowd'])
                            area.append(prop.area)

                            pil_image = PIL.Image.fromarray(binary_mask)
                            output_io = io.BytesIO()
                            pil_image.save(output_io, format='PNG')
                            encoded_mask_png.append(output_io.getvalue())

                        # Person Body
                        if np.count_nonzero(mask_body) != 0:
                            category_id = -2 
                            category_ids.append(category_id)
                            category_names.append('Person Body'.encode('utf8'))
                            
                            mask_body = cv2.resize(mask_body, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
                            mask_body = (mask_body == 1)
                            binary_mask = np.zeros((img_face.shape[0], img_face.shape[1]))
                            binary_mask[y1:y2, x1:x2][mask_body] = 1
                            binary_mask = np.uint8(binary_mask)
                            binary_mask = np.fliplr(binary_mask)    # Person Body Horizontal Flip

                            props = regionprops(binary_mask)
                            prop = props[0]

                            xmin.append(float(prop.bbox[1]) / image_width)
                            xmax.append(float(prop.bbox[3]) / image_width)
                            ymin.append(float(prop.bbox[0]) / image_height)
                            ymax.append(float(prop.bbox[2]) / image_height)
                            is_crowd.append(ann['iscrowd'])
                            area.append(prop.area)

                            pil_image = PIL.Image.fromarray(binary_mask)
                            output_io = io.BytesIO()
                            pil_image.save(output_io, format='PNG')
                            encoded_mask_png.append(output_io.getvalue())

                dp_triggered = False

            except KeyError:
                pass
        elif category_id == 1 and not dp_triggered:
            continue
        else:
            xmax.append(1 - float(x) / image_width)  # Xmin Horizontal Flip --> Xmax
            xmin.append(1 - float(x + width) / image_width)   # Xmax Horizontal Flip --> Xmin
            ymin.append(float(y) / image_height)
            ymax.append(float(y + height) / image_height)
            is_crowd.append(object_annotations['iscrowd'])

            category_ids.append(category_id)
            category_names.append(category_index[category_id]['name'].encode('utf8'))
            area.append(object_annotations['area'])

            run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                image_height, image_width)
            binary_mask = mask.decode(run_len_encoding)

            if not object_annotations['iscrowd']:
                binary_mask = np.amax(binary_mask, axis=2)  # (H, W, 1) --> (H, W)

            binary_mask = np.fliplr(binary_mask)  # Mask Horizontal Flip
            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())

    # Label Remapping
    # Background --> 0
    remapping = {
        # Person --> Person Face & Person Body
        -1: 1,  # Person Face
        -2: 2,  # Person Body
        2: 3, # Bicycle
        3: 4, # Car -> Car
        6: 4, # Bus -> Car
        4: 5, # Motorbike
        5: 6, # Airplane
        9: 7, # Ship (Boat)
        16: 8, # Bird
        17: 9, # Cat
        18: 10, # Dog
        19: 11, # Horse
        21: 12  # Cow
    }

    category_ids = list(map(lambda x: remapping[x], category_ids))

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(r),
        'image/width':
            dataset_util.int64_feature(r),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/label_text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label_id':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
        'image/object/mask':
            dataset_util.bytes_list_feature(encoded_mask_png)
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped, len(category_ids)



def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index):
    """Converts image and annotations to a tf.Example proto.

    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      annotations_list:
        list of dicts with keys:
        [u'segmentation', u'area', u'iscrowd', u'image_id',
        u'bbox', u'category_id', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
      image_dir: directory containing the image files.
      category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    origin_image = image

    

    r = 320
    image = image.resize((r, r), PIL.Image.ANTIALIAS)
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='JPEG')
    encoded_jpg = bytes_io.getvalue()
    key = hashlib.sha256(encoded_jpg).hexdigest()

    # Label Remapping
    # Background --> 0
    remapping = {
        # Person --> Person Face & Person Body
        -1: 1,  # Person Face
        -2: 2,  # Person Body
        2: 3, # Bicycle
        3: 4, # Car -> Car
        6: 4, # Bus -> Car
        4: 5, # Motorbike
        5: 6, # Airplane
        9: 7, # Ship (Boat)
        16: 8, # Bird
        17: 9, # Cat
        18: 10, # Dog
        19: 11, # Horse
        21: 12  # Cow
    }

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    dp_triggered = True  # DensePose preprocessing just one time
    binary_masks = []

    for object_annotations in annotations_list:
        category_id = int(object_annotations['category_id'])

        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue

        if category_id == 1 and dp_triggered: # Person --> Person Face & Person Body
            try:
                img = dp_coco.loadImgs(image_id)[0]
                ann_ids = dp_coco.getAnnIds(imgIds=img['id'])
                anns = dp_coco.loadAnns(ann_ids)

                for ann in anns:
                    if 'dp_masks' in ann.keys():
                        mask_face, mask_body = GenDPMask(ann)
                        img_face = cv2.imread(f'{image_dir}/{filename}')
                        img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
                        img_body = img_face.copy()

                        bbr = np.array(ann['bbox']).astype(int)
                        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0]+bbr[2], bbr[1]+bbr[3]
                        x2 = min( [x2, img_face.shape[1]] )
                        y2 = min( [y2, img_face.shape[0]] )

                        # Person Face
                        if np.count_nonzero(mask_face) != 0:
                            category_id = -1 
                            category_ids.append(category_id)
                            category_names.append('Person Face'.encode('utf8'))

                            mask_face = cv2.resize(mask_face, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
                            mask_face = (mask_face == 1)
                            binary_mask = np.zeros((img_face.shape[0], img_face.shape[1]))
                            binary_mask[y1:y2, x1:x2][mask_face] = 1
                            binary_mask = np.uint8(binary_mask)
                            
                            props = regionprops(binary_mask)
                            prop = props[0]

                            xmin.append(float(prop.bbox[1]) / image_width)
                            xmax.append(float(prop.bbox[3]) / image_width)
                            ymin.append(float(prop.bbox[0]) / image_height)
                            ymax.append(float(prop.bbox[2]) / image_height)
                            is_crowd.append(ann['iscrowd'])
                            area.append(prop.area)

                            binary_mask[binary_mask > 0] = 2
                            binary_masks.append(binary_mask)
                            pil_image = PIL.Image.fromarray(binary_mask)
                            output_io = io.BytesIO()
                            pil_image.save(output_io, format='PNG')
                            encoded_mask_png.append(output_io.getvalue())

                        # Person Body
                        if np.count_nonzero(mask_body) != 0:
                            category_id = -2 
                            category_ids.append(category_id)
                            category_names.append('Person Body'.encode('utf8'))
                            
                            mask_body = cv2.resize(mask_body, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
                            mask_body = (mask_body == 1)
                            binary_mask = np.zeros((img_face.shape[0], img_face.shape[1]))
                            binary_mask[y1:y2, x1:x2][mask_body] = 1
                            binary_mask = np.uint8(binary_mask)
                            
                            props = regionprops(binary_mask)
                            prop = props[0]

                            xmin.append(float(prop.bbox[1]) / image_width)
                            xmax.append(float(prop.bbox[3]) / image_width)
                            ymin.append(float(prop.bbox[0]) / image_height)
                            ymax.append(float(prop.bbox[2]) / image_height)
                            is_crowd.append(ann['iscrowd'])
                            area.append(prop.area)

                            binary_mask[binary_mask > 0] = 1
                            binary_masks.append(binary_mask)
                            pil_image = PIL.Image.fromarray(binary_mask)
                            output_io = io.BytesIO()
                            pil_image.save(output_io, format='PNG')
                            encoded_mask_png.append(output_io.getvalue())

                dp_triggered = False

            except KeyError:
                pass
        elif category_id == 1 and not dp_triggered:
            continue
        else:
            xmin.append(float(x) / image_width)
            xmax.append(float(x + width) / image_width)
            ymin.append(float(y) / image_height)
            ymax.append(float(y + height) / image_height)
            is_crowd.append(object_annotations['iscrowd'])

            category_ids.append(category_id)
            category_names.append(category_index[category_id]['name'].encode('utf8'))
            area.append(object_annotations['area'])

            run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                image_height, image_width)
            binary_mask = mask.decode(run_len_encoding)

            if not object_annotations['iscrowd']:
                binary_mask = np.amax(binary_mask, axis=2)  # (H, W, 1) --> (H, W)

            binary_mask[binary_mask > 0] = remapping[category_id]
            binary_masks.append(binary_mask)
            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())

    category_ids = list(map(lambda x: remapping[x], category_ids))

    if len(category_ids) != 0:
        if image_dir == FLAGS.train_image_dir:
            origin_image.save(f'ground_truth/train/{filename}')
        elif image_dir == FLAGS.val_image_dir:
            origin_image.save(f'ground_truth/test/{filename}')

        binary_masks = np.stack(binary_masks, axis=-1)
        binary_masks = np.amax(binary_masks, axis=-1)
        binary_masks = binary_masks.astype(np.uint8)
        mask_PIL = PIL.Image.fromarray(binary_masks)

        if image_dir == FLAGS.train_image_dir:
            mask_PIL.save(f'ground_truth/train/{filename.split(".")[0]}_mask.png')
        elif image_dir == FLAGS.val_image_dir:
            mask_PIL.save(f'ground_truth/test/{filename.split(".")[0]}_mask.png')

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(r),
        'image/width':
            dataset_util.int64_feature(r),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/label_text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label_id':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
        'image/object/mask':
            dataset_util.bytes_list_feature(encoded_mask_png)
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped, len(category_ids)


def _create_tf_record_from_coco_annotations(
        annotations_file, image_dir, output_path, num_shards):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
      annotations_file: JSON file containing bounding box annotations.
      image_dir: Directory containing the image files.
      output_path: Path to output tf.Record file.
      num_shards: number of output file shards.
    """
    with contextlib2.ExitStack() as tf_record_close_stack, \
            tf.io.gfile.GFile(annotations_file, 'r') as fid:
        output_tfrecords = dataset_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        category_index = dataset_util.create_category_index(
            groundtruth_data['categories'])

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            logging.info(
                'Found groundtruth annotations. Building annotations index.')
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)
        
        missing_annotation_count = 0
        
        for image in images:
            image_id = image['id']
            if image_id not in annotations_index:
                missing_annotation_count += 1
                annotations_index[image_id] = []
        
        logging.info('%d images are missing annotations.',
                     missing_annotation_count)

        total_num_annotations_skipped = 0
        total_num_instances = 0

        for idx, image in enumerate(images):
            if idx % 100 == 0:
                logging.info(f'On image {idx} of {len(images)}. Process {total_num_instances} instances.')
            annotations_list = annotations_index[image['id']]
            annotations_list = list(filter(dataset_util.lg_filter, annotations_list))

            # ignore image only have crowd annotation
            num_crowd = 0
            for object_annotations in annotations_list:
                if object_annotations['iscrowd']:
                    num_crowd += 1
            
            if num_crowd != len(annotations_list):
                _, tf_example, num_annotations_skipped, num_instances = create_tf_example(image, annotations_list, image_dir, category_index)
                if num_instances != 0:
                    total_num_annotations_skipped += num_annotations_skipped
                    total_num_instances += num_instances
                    shard_idx = idx % num_shards
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())

                _, tf_example, num_annotations_skipped, num_instances = create_tf_horizontal_flip_example(image, annotations_list, image_dir, category_index)
                if num_instances != 0:
                    total_num_annotations_skipped += num_annotations_skipped
                    total_num_instances += num_instances
                    shard_idx = idx % num_shards
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            else:
                total_num_annotations_skipped += len(annotations_list)

        logging.info('Finished writing, skipped %d annotations.',
                     total_num_annotations_skipped)


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    # assert FLAGS.test_image_dir, '`test_image_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
    # assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'

    if not tf.io.gfile.isdir(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir, 'coco_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'coco_val.record')
    # testdev_output_path = os.path.join(FLAGS.output_dir, 'coco_testdev.record')

    _create_tf_record_from_coco_annotations(
        FLAGS.train_annotations_file,
        FLAGS.train_image_dir,
        train_output_path,
        num_shards=100)
    _create_tf_record_from_coco_annotations(
        FLAGS.val_annotations_file,
        FLAGS.val_image_dir,
        val_output_path,
        num_shards=10)
    """
    _create_tf_record_from_coco_annotations(
        FLAGS.testdev_annotations_file,
        FLAGS.test_image_dir,
        testdev_output_path,
        num_shards=100)
    """


if __name__ == '__main__':
    app.run(main)
