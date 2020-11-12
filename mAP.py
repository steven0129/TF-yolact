import tensorflow as tf
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import lite
import numpy as np
import csv

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
        cur_score = cls_pred[batch_idx, 1:, :]
        conf_score = tf.math.reduce_max(cur_score, axis=0)
        conf_score_id = tf.argmax(cur_score, axis=0)

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

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

class PRCurve():
    def __init__(self, prediction, ground_truth, sz_mode='all'):
        self.label_mapping = [
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
    
        if sz_mode == 'all':
            self.prediction = prediction
            self.ground_truth = ground_truth
        elif sz_mode == 'small':   # < 32 * 32
            self.prediction = prediction
            self.ground_truth = list(filter(lambda x: ((x['bbox'][1] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][2]) < (32 * 32)), ground_truth))
        elif sz_mode == 'medium':  # 32 * 32 ~ 96 * 96
            self.prediction = prediction
            self.ground_truth = list(filter(lambda x: ((x['bbox'][1] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][2]) >= (32 * 32)) and ((x['bbox'][1] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][2]) < (96 * 96)), ground_truth))
        elif sz_mode == 'large':   # > 96 * 96
            self.prediction = prediction
            self.ground_truth = list(filter(lambda x: ((x['bbox'][1] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][2]) >= (96 * 96)), ground_truth))

    def run(self, cls_idx, iou_threshold=0.5, use_maskiou=False):
        TP = 0
        FP = 0
        precisions = []
        recalls = []
        
        gt = list(filter(lambda x: x['class'] == cls_idx, self.ground_truth))
        pred = list(filter(lambda x: x['class'] == cls_idx, self.prediction))
        pred = sorted(pred, key=lambda x: x['score'], reverse=True)

        total_gt_num = len(gt)

        if len(gt) != 0:
            for pidx, p in enumerate(pred):
                iou = []
                img_idx = p['img_idx']
                img = cv2.imread(p['origin'])
                filename = p['filename']
                
                for g in filter(lambda x: x['img_idx'] == img_idx, gt):
                    iou.append((utils.jaccard_numpy(p['bbox'], g['bbox']), g))

                filtered_iou = list(filter(lambda x: x[0] > iou_threshold, iou))
                if(len(filtered_iou) != 0):
                    max_iou = max(filtered_iou, key=lambda x: x[0])
                    if max_iou[0] < iou_threshold + 0.1:
                        g = max_iou[1]
                        iou_file_folder = f'AP/{self.label_mapping[cls_idx]}/iou_{iou_threshold}-{iou_threshold + 0.1}'
                        if not os.path.isdir(iou_file_folder):
                            os.mkdir(iou_file_folder)
                        
                        # Draw ground truth
                        cv2.rectangle(img, (g['bbox'][1], g['bbox'][0]), (g['bbox'][3], g['bbox'][2]), (0, 0, 255), 2)
                        cv2.putText(img, self.label_mapping[g['class']], (int(g['bbox'][1]), int(g['bbox'][0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                        
                        # Draw prediction
                        cv2.rectangle(img, (p['bbox'][1], p['bbox'][0]), (p['bbox'][3], p['bbox'][2]), (255, 0, 0), 2)
                        cv2.putText(img, self.label_mapping[p['class']], (int(p['bbox'][1]), int(p['bbox'][0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                        
                        # Draw segmentation
                        mask = p['mask'].astype(np.uint8)
                        mask[mask > 0] = p['class']
                        mask = mask / 13 * 255
                        mask = 255 - mask

                        # Save image
                        cv2.imwrite(f'{iou_file_folder}/{filename.split(".")[0]}_{pidx}_mask.png', mask)
                        cv2.imwrite(f'{iou_file_folder}/{filename.split(".")[0]}_{pidx}_box.png', img)
                    
                    TP += 1
                else:
                    FP += 1

                precisions.append(TP / (pidx + 1))
                recalls.append(TP / total_gt_num)

        return recalls, precisions

    def _maskiou(self, pred_mask, gt_mask):
        gt_mask = np.resize(gt_mask, pred_mask.shape)
        gt_mask = np.reshape(gt_mask, (-1)).astype(np.float32)
        pred_mask = np.reshape(pred_mask, (-1)).astype(np.float32)
        
        gt_area = np.sum(gt_mask, axis=0)
        pred_area = np.sum(pred_mask, axis=0)
        intersection = np.dot(gt_mask.T, pred_mask)

        if (gt_area + pred_area - intersection) != 0:
            return intersection / (gt_area + pred_area - intersection)
        else:
            return -1
        
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

ckpt_dir = "checkpoints-SGD-20201019"
latest = tf.train.latest_checkpoint(ckpt_dir)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
print("Restore Ckpt Sucessfully!!")

# Load Validation Images and do Detection
# -----------------------------------------------------------------------------------------------
# Need default anchor
anchorobj = anchor.Anchor(img_size=256, feature_map_size=[32, 16, 8, 4, 2], aspect_ratio=[1, 0.5, 2], scale=[24, 48, 96, 192, 384])
valid_dataset = dataset_coco.prepare_evalloader(img_size=256,
                                                tfrecord_dir='data/obj_tfrecord_256x256_20200930',
                                                subset='val')
anchors = anchorobj.get_anchors()
detect_layer = Detect(num_cls=13, label_background=0, top_k=200, conf_threshold=0.4, nms_threshold=0.5, anchors=anchors)

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

confusion_matrix = [[0 for _ in range(13)] for j in range(13)]

ground_truth = []
prediction = []

for img_idx, (image, labels) in enumerate(tqdm(valid_dataset)):
    # only try on 1 image
    output = model(image, training=False)
    detection = detect_layer(output)

    filename = labels['filename'].numpy().tolist()[0].decode('utf-8')
    gt_bbox = labels['bbox'].numpy()
    gt_mask = labels['mask_target'].numpy().astype(int)
    gt_cls = labels['classes'].numpy()
    num_obj = labels['num_obj'].numpy()

    if detection[0]['detection'] != None:
        my_cls, scores, bbox, masks = postprocess(detection, 256, 256, 0, 'bilinear')
        my_cls, scores, bbox, masks = my_cls.numpy(), scores.numpy(), bbox.numpy(), masks.numpy()
        
        for label_name in remapping:
            if not os.path.isdir(f'AP/{label_name}') and not label_name == 'Background':
                os.mkdir(f'AP/{label_name}')
                os.mkdir(f'AP/{label_name}/origin')

        for idx in range(num_obj[0]):
            gt_cls_idx = gt_cls[0][idx]
            gt_cls_name = remapping[gt_cls[0][idx]]

            AP_file_path = f'AP/{gt_cls_name}/origin/{filename.split(".")[0]}.png'
            deimage = denormalize_image(image)
            deimage = tf.squeeze(deimage).numpy()
            deimage = cv2.cvtColor(deimage, cv2.COLOR_BGR2RGB)
            cv2.imwrite(AP_file_path, deimage)
            
            ground_truth.append({
                'img_idx': img_idx,
                'filename': filename,
                'origin': AP_file_path,
                'class': gt_cls_idx,
                'mask': gt_mask[0][idx],
                'bbox': gt_bbox[0][idx],
            })

        for idx in range(bbox.shape[0]):
            prediction.append({
                'img_idx': img_idx,
                'filename': filename,
                'origin': AP_file_path,
                'class': my_cls[idx] + 1,
                'score': scores[idx],
                'mask': masks[idx],
                'bbox': bbox[idx]
            })

for iou_type in tqdm(['box']):
    for sz_mode in tqdm(['all']):
        PRObj = PRCurve(prediction, ground_truth, sz_mode=sz_mode)

        with open(f'AP/AP_{iou_type}_{sz_mode}.csv', 'w') as FILE:
            writer = csv.writer(FILE)
            iou_ths = [i * 0.01 for i in range(50, 96, 5)]
            writer.writerow(['category'] + list(map(lambda x: f'AP@{x:.2f}', iou_ths)) + ['AP@0.5:0.05:0.95'])
            
            for cls_idx in tqdm(range(1, 13)):  # Process class by class
                cls_name = remapping[cls_idx]
                APs = []

                for iou_threshold in iou_ths:
                    if iou_type == 'box':
                        recalls, precisions = PRObj.run(cls_idx, iou_threshold, use_maskiou=False)
                    elif iou_type == 'mask':
                        recalls, precisions = PRObj.run(cls_idx, iou_threshold, use_maskiou=True)

                    prev_recall = 0
                    AP = 0
                    for i in range(1, len(recalls)):
                        AP += precisions[i] * (recalls[i] - prev_recall)
                        prev_recall = recalls[i]

                    APs.append(AP * 100)

                    if iou_threshold == 0.5:
                        plt_line = plt.plot(recalls, precisions, linewidth=0.5, label=cls_name)
                
                AP_avg = sum(APs) / len(APs)
                writer.writerow([cls_name] + list(map(lambda x: f'{x:.5f}', APs)) + [AP_avg])

            plt.legend(prop={'size': 6})
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.savefig(f'AP/PR_curve_{iou_type}_{sz_mode}_threshold50.png')
            plt.clf()