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
        objectness = tf.math.sigmoid(cls_pred[batch_idx, :, 0])
        classification = tf.nn.softmax(cls_pred[batch_idx, :, 1:], axis=-1)

        conf_score = tf.math.reduce_max(classification, axis=-1)
        conf_score_id = tf.argmax(classification, axis=-1)
        tf.print("conf_score:", tf.shape(conf_score))
        tf.print(f'conf_score_id: {tf.math.bincount(tf.cast(conf_score_id, dtype=tf.int32))}')

        # filter out the ROI that have conf score > confidence threshold
        test_indices = tf.where(objectness > 0.45)
        test_objectness = tf.gather(objectness, test_indices)
        tf.print(test_objectness)
        candidate_ROI_idx = tf.squeeze(tf.where(tf.logical_and(objectness > 0.45, conf_score > self.conf_threshold)))
        tf.print("candidate_ROI", tf.shape(candidate_ROI_idx))

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

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

lr_schedule = learning_rate_schedule.Yolact_LearningRateSchedule(warmup_steps=500, warmup_lr=1e-4, initial_lr=1e-3)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

YOLACT = lite.MyYolact(input_size=256,
               fpn_channels=96,
               feature_map_size=[32, 16, 8, 4, 2],
               num_class=13,
               num_mask=32,
               aspect_ratio=[1, 0.5, 2],
               scales=[24, 48, 96, 192, 384])

model = YOLACT.gen()

ckpt_dir = "checkpoints-SGD"
latest = tf.train.latest_checkpoint(ckpt_dir)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

# Load Validation Images and do Detection
# -----------------------------------------------------------------------------------------------
# Need default anchor
anchorobj = anchor.Anchor(img_size=256, feature_map_size=[32, 16, 8, 4, 2], aspect_ratio=[1, 0.5, 2], scale=[24, 48, 96, 192, 384])
valid_dataset = dataset_coco.prepare_dataloader(img_size=256,
                                                tfrecord_dir='data/obj_tfrecord_256x256_20201102',
                                                batch_size=1,
                                                subset='val')
anchors = anchorobj.get_anchors()
detect_layer = Detect(num_cls=13, label_background=0, top_k=200, conf_threshold=0.7, nms_threshold=0.5, anchors=anchors)

for image, labels in valid_dataset.take(1):
    tf.print( 'classes', tf.boolean_mask(labels['classes'], labels['classes'] > 0) )
    # only try on 1 image
    output = model(image, training=False)
    detection = detect_layer(output)
    dets = postprocess(detection, 256, 256, 0, 'bilinear')

    if dets != None:
        my_cls, scores, bbox, masks = dets

        tf.print(f'cls: {tf.shape(my_cls)}')
        tf.print(f'scores: {scores}')
        tf.print(f'bbox: {tf.shape(bbox)}')
        tf.print(f'masks: {tf.shape(masks)}')

        my_cls, scores, bbox, masks = my_cls.numpy(), scores.numpy(), bbox.numpy(), masks.numpy()
        image = denormalize_image(image)
        
        image = tf.squeeze(image).numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('origin.png', image)

        gt_bbox = labels['bbox'].numpy()
        gt_cls = labels['classes'].numpy()
        num_obj = labels['num_obj'].numpy()

        # show ground truth
        print('===============Label=============')
        for idx in range(num_obj[0]):
            b = gt_bbox[0][idx]
            cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), (0, 0, 255), 2)

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

            print(remapping[gt_cls[0][idx]])
            cv2.putText(image, remapping[gt_cls[0][idx]], (int(b[1]), int(b[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 255), 1)

        # show the prediction box
        print()
        print('===============Prediction=============')
        for idx in range(bbox.shape[0]):
            b = bbox[idx]
            score = scores[idx]
            cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), (255, 0, 0), 2)

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


            print(remapping[my_cls[idx]+1])
            cv2.putText(image, f'{remapping[my_cls[idx]+1]} {score: .3f}', (int(b[1]), int(b[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        cv2.imwrite('result.png', image)

        # show the mask
        seg = np.zeros([256, 256, masks.shape[0]])

        for idx in range(masks.shape[0]):
            mask = masks[idx].astype(np.uint8)
            mask[mask > 0] = my_cls[idx]+1
            seg[:, :, idx] = mask

        seg = seg / 13 * 255
        seg = np.amax(seg, axis=-1)
        seg = seg.astype(np.uint8)
        seg = 255 - seg
        cv2.imwrite('seg.png', seg)
    else:
        image = denormalize_image(image)
        image = tf.squeeze(image).numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_bbox = labels['bbox'].numpy()
        gt_cls = labels['classes'].numpy()
        num_obj = labels['num_obj'].numpy()

        # show ground truth
        print('===============Label=============')
        for idx in range(num_obj[0]):
            b = gt_bbox[0][idx]
            cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), (0, 0, 255), 2)

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
            print(remapping[gt_cls[0][idx]])
            cv2.putText(image, remapping[gt_cls[0][idx]], (int(b[1]), int(b[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 255), 1)
        
        print()
        print('===============Prediction===========')
        cv2.imwrite('none.png', image)
        print('None of object is detected.')
