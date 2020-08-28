import tensorflow as tf
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import lite
import numpy as np

from data import dataset_coco, anchor
from utils import learning_rate_schedule, label_map
from yolact import Yolact
from utils.utils import postprocess, denormalize_image
from utils import utils
from tqdm import tqdm

import cv2

class Detect(object):
    def __init__(self, num_cls, label_background, top_k, conf_threshold, nms_threshold, anchors):
        self.num_cls = num_cls
        self.label_background = label_background
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.anchors = anchors

    def __call__(self, model_output):
        prediction = {
            'proto_out': model_output[1],
            'pred_cls': model_output[2],
            'pred_offset': model_output[3],
            'pred_mask_coef': model_output[4]
        }

        loc_pred = prediction['pred_offset']
        cls_pred = prediction['pred_cls']
        mask_pred = prediction['pred_mask_coef']
        proto_pred = prediction['proto_out']
        out = []
        num_batch = tf.shape(loc_pred)[0]
        num_anchors = tf.shape(loc_pred)[1]

        # apply softmax to pred_cls
        cls_pred = tf.nn.softmax(cls_pred, axis=-1)
        cls_pred = tf.transpose(cls_pred, perm=[0, 2, 1])

        for batch_idx in tf.range(num_batch):
            # add offset to anchors
            decoded_boxes = utils.map_to_bbox(self.anchors, loc_pred[batch_idx])
            # do detection
            result = self._detection(batch_idx, cls_pred, decoded_boxes, mask_pred)
            if result is not None and proto_pred is not None:
                result['proto'] = proto_pred[batch_idx]
            out.append({'detection': result})

        return out

    def _detection(self, batch_idx, cls_pred, decoded_boxes, mask_pred):
        # we don't need to deal with background label
        cur_score = cls_pred[batch_idx, 1:, :]
        conf_score = tf.math.reduce_max(cur_score, axis=0)
        conf_score_id = tf.argmax(cur_score, axis=0)
        # tf.print(tf.math.bincount(conf_score_id, dtype=tf.dtypes.int64))

        # filter out the ROI that have conf score > confidence threshold
        candidate_ROI_idx = tf.squeeze(tf.where(conf_score > self.conf_threshold))

        if tf.size(candidate_ROI_idx) == 0:
            return None

        # scores = tf.gather(cur_score, candidate_ROI_idx, axis=-1)
        scores = tf.gather(conf_score, candidate_ROI_idx)
        classes = tf.gather(conf_score_id, candidate_ROI_idx)
        boxes = tf.gather(decoded_boxes, candidate_ROI_idx)
        masks = tf.gather(mask_pred[batch_idx], candidate_ROI_idx)

        if tf.shape(tf.shape(boxes))[0] == 1:
            scores = tf.expand_dims(scores, axis=0)
            boxes = tf.expand_dims(boxes, axis=0)
            masks = tf.expand_dims(masks, axis=0)
            classes = tf.expand_dims(classes, axis=0)

            return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

        selected_indices = tf.image.non_max_suppression(boxes, scores, 100, self.nms_threshold)

        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(scores, selected_indices)
        masks = tf.gather(masks, selected_indices)
        classes = tf.gather(classes, selected_indices)

        # apply fast nms for final detection
        # top_k = tf.math.minimum(self.top_k, tf.size(candidate_ROI_idx))
        # boxes, masks, classes, scores = self._fast_nms(boxes, masks, scores, self.nms_threshold, top_k)

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def _fast_nms(self, boxes, masks, scores, iou_threshold=0.5, top_k=200, second_threshold=False):
        scores, idx = tf.math.top_k(scores, k=top_k)
        num_classes, num_dets = tf.shape(idx)[0], tf.shape(idx)[1]

        boxes = tf.gather(boxes, idx)

        masks = tf.gather(masks, idx)

        iou = utils.jaccard(boxes, boxes)

        # upper trangular matrix - diagnoal
        upper_triangular = tf.linalg.band_part(iou, 0, -1)
        diag = tf.linalg.band_part(iou, 0, 0)
        iou = upper_triangular - diag

        # fitler out the unwanted ROI
        iou_max = tf.reduce_max(iou, axis=1)
        idx_det = tf.where(iou_max < iou_threshold)

        classes = tf.broadcast_to(tf.expand_dims(tf.range(num_classes), axis=-1), tf.shape(iou_max))
        classes = tf.gather_nd(classes, idx_det)
        boxes = tf.gather_nd(boxes, idx_det)
        masks = tf.gather_nd(masks, idx_det)
        scores = tf.gather_nd(scores, idx_det)

        max_num_detection = tf.math.minimum(self.top_k, tf.size(scores))
        # number of max detection = 100 (u can choose whatever u want)
        scores, idx = tf.math.top_k(scores, k=max_num_detection)
        classes = tf.gather(classes, idx)
        boxes = tf.gather(boxes, idx)
        masks = tf.gather(masks, idx)
        scores = tf.gather(scores, idx)

        # Todo Handle the situation that only 1 or 0 detection
        # second threshold
        positive_det = tf.squeeze(tf.where(scores > self.conf_threshold))
        scores = tf.gather(scores, positive_det)
        classes = classes[:tf.size(scores)]
        boxes = boxes[:tf.size(scores)]
        masks = masks[:tf.size(scores)]

        return boxes, masks, classes, scores

lr_schedule = learning_rate_schedule.Yolact_LearningRateSchedule(warmup_steps=500, warmup_lr=1e-4, initial_lr=1e-3)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

YOLACT = lite.MyYolact(input_size=320,
               fpn_channels=256,
               feature_map_size=[40, 20, 10, 5, 3],
               num_class=13,
               num_mask=32,
               aspect_ratio=[1, 0.5, 2],
               scales=[24, 48, 96, 192, 384])

model = YOLACT.gen()

ckpt_dir = "checkpoints-SGD"
latest = tf.train.latest_checkpoint(ckpt_dir)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
print("Restore Ckpt Sucessfully!!")

# Load Validation Images and do Detection
# -----------------------------------------------------------------------------------------------
# Need default anchor
anchorobj = anchor.Anchor(img_size=320, feature_map_size=[40, 20, 10, 5, 3], aspect_ratio=[1, 0.5, 2], scale=[24, 48, 96, 192, 384])
valid_dataset = dataset_coco.prepare_dataloader(img_size=320,
                                                tfrecord_dir='data/coco_tfrecord_320x320_20200826',
                                                batch_size=1,
                                                subset='val')
anchors = anchorobj.get_anchors()
detect_layer = Detect(num_cls=13, label_background=0, top_k=200, conf_threshold=0.5, nms_threshold=0.5, anchors=anchors)

remapping = [
    'Background',
    'Face',
    'Body',
    'Bicycle',
    'Car',
    'Motorbike',
    'Airplane',
    'Ship',
    'Bird',
    'Cat',
    'Dog',
    'Horse',
    'Cow'
]

TT = 0  # 應被detect但被detect
TF = 0  # 應被detect但沒被detect
FT = 0  # 應沒被detect但被detect
total = 0

for image, labels in tqdm(valid_dataset.take(3000)):
    # only try on 1 image
    output = model(image, training=False)
    detection = detect_layer(output)

    gt_cls = labels['classes'].numpy()
    num_obj = labels['num_obj'].numpy()

    if detection[0]['detection'] != None:
        my_cls, scores, bbox, masks = postprocess(detection, 320, 320, 0, 'bilinear')
        my_cls, scores, bbox, masks = my_cls.numpy(), scores.numpy(), bbox.numpy(), masks.numpy()
        ground_truth = []
        prediction = []

        for idx in range(num_obj[0]):
            ground_truth.append(gt_cls[0][idx])

        for idx in range(bbox.shape[0]):
            prediction.append(my_cls[idx]+1)

        # TT
        gt_for_tt = ground_truth.copy()
        pred_for_tt = prediction.copy()
        for pred in prediction:
            if pred in gt_for_tt:
                TT += 1
                gt_for_tt.remove(pred)
                pred_for_tt.remove(pred)

        TF += len(gt_for_tt)
        FT += len(pred_for_tt)

    else:
        TF += num_obj[0]

    tqdm.write(f'TT: {TT}')
    tqdm.write(f'TF: {TF}')
    tqdm.write(f'FT: {FT}')

print('TT:', TT)
print('TF:', TF)
print('FT:', FT)