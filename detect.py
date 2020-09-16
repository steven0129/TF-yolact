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
        tf.print('loc pred', tf.shape(loc_pred))
        cls_pred = prediction['pred_cls']
        tf.print('cls pred', tf.shape(cls_pred))
        mask_pred = prediction['pred_mask_coef']
        tf.print('mask pred', tf.shape(mask_pred))
        proto_pred = prediction['proto_out']
        tf.print('proto pred', tf.shape(proto_pred))
        tf.print('anchors', tf.shape(self.anchors))
        out = []
        num_batch = tf.shape(loc_pred)[0]
        num_anchors = tf.shape(loc_pred)[1]
        tf.print("num batch:", num_batch)
        tf.print("num anchors:", num_anchors)

        # apply softmax to pred_cls
        cls_pred = tf.nn.softmax(cls_pred, axis=-1)
        tf.print("score", tf.shape(cls_pred))
        cls_pred = tf.transpose(cls_pred, perm=[0, 2, 1])
        tf.print("score", tf.shape(cls_pred))

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
        tf.print(f'cls_pred: {tf.shape(cls_pred)}')
        cur_score = cls_pred[batch_idx, 1:, :]
        tf.print("cur score:", tf.shape(cur_score))
        conf_score = tf.math.reduce_max(cur_score, axis=0)
        conf_score_id = tf.argmax(cur_score, axis=0)
        tf.print("conf_score:", tf.shape(conf_score))
        tf.print(f'conf_score_id: {tf.math.bincount(tf.cast(conf_score_id, dtype=tf.int32))}')
        # tf.print(tf.math.bincount(conf_score_id, dtype=tf.dtypes.int64))

        # filter out the ROI that have conf score > confidence threshold
        tf.print('highest score', tf.reduce_max(conf_score))
        candidate_ROI_idx = tf.squeeze(tf.where(conf_score > self.conf_threshold))
        tf.print("candidate_ROI", tf.shape(candidate_ROI_idx))

        if tf.size(candidate_ROI_idx) == 0:
            return None

        # scores = tf.gather(cur_score, candidate_ROI_idx, axis=-1)
        scores = tf.gather(conf_score, candidate_ROI_idx)
        classes = tf.gather(conf_score_id, candidate_ROI_idx)
        tf.print("scores", tf.shape(scores))
        boxes = tf.gather(decoded_boxes, candidate_ROI_idx)
        tf.print("boxes", tf.shape(boxes))
        masks = tf.gather(mask_pred[batch_idx], candidate_ROI_idx)
        tf.print("masks", tf.shape(masks))

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
        tf.print("top k scores:", tf.shape(scores))
        tf.print("top k indices", tf.shape(idx))
        tf.print(idx[:20])

        num_classes, num_dets = tf.shape(idx)[0], tf.shape(idx)[1]
        tf.print("num_classes:", num_classes)
        tf.print("num dets:", num_dets)

        tf.print("old boxes", tf.shape(boxes))

        boxes = tf.gather(boxes, idx)
        tf.print("new boxes", tf.shape(boxes))

        masks = tf.gather(masks, idx)
        tf.print("new masks", tf.shape(masks))

        iou = utils.jaccard(boxes, boxes)
        tf.print("iou", tf.shape(iou))

        # upper trangular matrix - diagnoal
        upper_triangular = tf.linalg.band_part(iou, 0, -1)
        diag = tf.linalg.band_part(iou, 0, 0)
        tf.print("upper tri", upper_triangular[0])
        iou = upper_triangular - diag
        tf.print("iou", tf.shape(iou))
        tf.print("iou", iou[0])

        # fitler out the unwanted ROI
        iou_max = tf.reduce_max(iou, axis=1)
        tf.print("iou max", tf.shape(iou_max))
        tf.print("iou max", iou_max)

        idx_det = tf.where(iou_max < iou_threshold)

        tf.print("idx det", tf.shape(idx_det))
        tf.print(idx_det)

        classes = tf.broadcast_to(tf.expand_dims(tf.range(num_classes), axis=-1), tf.shape(iou_max))
        tf.print("classes", classes)
        classes = tf.gather_nd(classes, idx_det)
        tf.print("new_classes", tf.shape(classes))
        tf.print(classes)
        boxes = tf.gather_nd(boxes, idx_det)
        tf.print("new_boxes", tf.shape(boxes))
        masks = tf.gather_nd(masks, idx_det)
        tf.print("new_masks", tf.shape(masks))
        scores = tf.gather_nd(scores, idx_det)
        tf.print("new_scores", tf.shape(scores))
        tf.print(scores)

        max_num_detection = tf.math.minimum(self.top_k, tf.size(scores))
        # number of max detection = 100 (u can choose whatever u want)
        scores, idx = tf.math.top_k(scores, k=max_num_detection)
        tf.print("max num score", scores)
        classes = tf.gather(classes, idx)
        tf.print("max num classes", classes)
        boxes = tf.gather(boxes, idx)
        masks = tf.gather(masks, idx)
        scores = tf.gather(scores, idx)

        # Todo Handle the situation that only 1 or 0 detection
        # second threshold
        positive_det = tf.squeeze(tf.where(scores > self.conf_threshold))
        scores = tf.gather(scores, positive_det)
        classes = classes[:tf.size(scores)]
        tf.print("final classes", classes)
        boxes = boxes[:tf.size(scores)]
        masks = masks[:tf.size(scores)]

        tf.print("final score", scores)
        tf.print("num_final_detection", tf.size(scores))

        return boxes, masks, classes, scores

lr_schedule = learning_rate_schedule.Yolact_LearningRateSchedule(warmup_steps=500, warmup_lr=1e-4, initial_lr=1e-3)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

YOLACT = lite.MyYolact(input_size=256,
               fpn_channels=128,
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
print("Restore Ckpt Sucessfully!!")

# Load Validation Images and do Detection
# -----------------------------------------------------------------------------------------------
# Need default anchor
anchorobj = anchor.Anchor(img_size=256, feature_map_size=[32, 16, 8, 4, 2], aspect_ratio=[1, 0.5, 2], scale=[24, 48, 96, 192, 384])
valid_dataset = dataset_coco.prepare_dataloader(img_size=256,
                                                tfrecord_dir='data/obj_tfrecord_256x256_20200916',
                                                batch_size=1,
                                                subset='val')
anchors = anchorobj.get_anchors()
tf.print(tf.shape(anchors))
detect_layer = Detect(num_cls=13, label_background=0, top_k=200, conf_threshold=0.3, nms_threshold=0.5, anchors=anchors)

for image, labels in valid_dataset.take(1):
    print('mask_target', np.count_nonzero(labels['mask_target']))
    # only try on 1 image
    output = model(image, training=False)
    detection = detect_layer(output)
    print(len(detection))

    my_cls, scores, bbox, masks = postprocess(detection, 256, 256, 0, 'bilinear')

    tf.print(f'cls: {tf.shape(my_cls)}')
    tf.print(f'scores: {tf.shape(scores)}')
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
        cv2.putText(image, remapping[gt_cls[0][idx]], (int(b[1]), int(b[0]) - 10), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, (0, 0, 255), 1)

    print('---------------------')
    # show the prediction box
    for idx in range(bbox.shape[0]):
        b = bbox[idx]
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
        cv2.putText(image, remapping[my_cls[idx]+1], (int(b[1]), int(b[0]) - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)

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
