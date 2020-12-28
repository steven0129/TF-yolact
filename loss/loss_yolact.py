import tensorflow as tf
import time
import tensorflow_addons as tfa
from utils import utils
from data import anchor

class GHM_Loss():
    def __init__(self, bins=10, momentum=0.75):
        self.g = None
        self.bins = bins
        self.momentum = momentum
        self.valid_bins = tf.constant(0.0, dtype=tf.float32)
        self.edges_left, self.edges_right = self.get_edges(self.bins)
        if momentum > 0:
            acc_sum = [0.0 for _ in range(bins)]
            self.acc_sum = tf.Variable(acc_sum, trainable=False)

    @staticmethod
    def get_edges(bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left)  # [bins]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-3
        edges_right = tf.constant(edges_right)  # [bins]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1, 1]
        return edges_left, edges_right

    def calc(self, g, valid_mask):
        edges_left, edges_right = self.edges_left, self.edges_right
        alpha = self.momentum
        # valid_mask = tf.cast(valid_mask, dtype=tf.bool)

        tot = tf.maximum(tf.reduce_sum(tf.cast(valid_mask, dtype=tf.float32)), 1.0)

        # g --> [1, batch_num, class_num]
        # edges_left --> [bins, 1, 1, 1]
        inds_mask = tf.logical_and(g >= edges_left, g < edges_right)  # [bins, 1, batch_num, class_num]
        zero_matrix = tf.cast(tf.zeros_like(inds_mask), dtype=tf.float32)  # [bins, 1, batch_num, class_num]

        inds = tf.cast(tf.logical_and(inds_mask, valid_mask), dtype=tf.float32)  # [bins, 1, batch_num, class_num]
        num_in_bin = tf.reduce_sum(inds, axis=[1, 2, 3])  # [bins]
        valid_bins = tf.greater(num_in_bin, 0)  # [bins]

        num_valid_bin = tf.reduce_sum(tf.cast(valid_bins, dtype=tf.float32))

        if alpha > 0:
            self.acc_sum = tf.where(valid_bins, alpha * self.acc_sum + (1 - alpha) * num_in_bin, self.acc_sum)
            acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
            acc_sum = tf.expand_dims(acc_sum_tmp, -1)  # [bins, 1]
            acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
            acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1, 1]
            acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)
        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1, 1]
            num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)

        weights = weights / num_valid_bin
        return weights, tot

    def class_loss(self, logits, targets, masks=None):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """

        train_mask = (1 - tf.cast(tf.equal(targets, -1), dtype=tf.float32))
        self.g = tf.abs(tf.sigmoid(logits) - targets) # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0)  # [1, batch_num, class_num]

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0
        weights, tot = self.calc(g, valid_mask)
        ghm_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets*train_mask, logits=logits)
        ghm_class_loss = tf.reduce_sum(ghm_class_loss * weights) / tot

        return ghm_class_loss

    def regression_loss(self, logits, targets, masks):
        """ Args:
        input [batch_num, *(* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num,  *(* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu

        # ASL1 loss
        diff = logits - targets
        # gradient length
        g = tf.abs(diff / tf.sqrt(mu * mu + diff * diff))

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0

        weights, tot = self.calc(g, valid_mask)

        ghm_reg_loss = tf.sqrt(diff * diff + mu * mu) - mu
        ghm_reg_loss = tf.reduce_sum(ghm_reg_loss * weights) / tot

        return ghm_reg_loss



class YOLACTLoss(object):

    def __init__(self, loss_weight_cls=1,
                 loss_weight_box=1.5,
                 loss_weight_mask=6.125,
                 loss_seg=1,
                 neg_pos_ratio=3,
                 max_masks_for_train=100):
        self._loss_weight_cls = loss_weight_cls
        self._loss_weight_box = loss_weight_box
        self._loss_weight_mask = loss_weight_mask
        self._loss_weight_seg = loss_seg
        self._neg_pos_ratio = neg_pos_ratio
        self._max_masks_for_train = max_masks_for_train
        self.focal_loss_with_logits = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, gamma=10)
        self.giou_loss = tfa.losses.GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
        self.anchors = anchor.Anchor(img_size=550, feature_map_size=[69, 35, 18, 9, 5])

    def __call__(self, pred, label, num_classes):
        """
        :param num_classes:
        :param anchors:
        :param label: labels dict from dataset
        :param pred:
        :return:
        """
        # all prediction component
        pred_cls = pred['pred_cls']
        pred_offset = pred['pred_offset']
        pred_mask_coef = pred['pred_mask_coef']
        proto_out = pred['proto_out']
        seg = pred['seg']

        # all label component
        cls_targets = label['cls_targets']
        box_targets = label['box_targets']
        positiveness = label['positiveness']
        bbox_norm = label['bbox_for_norm']
        gt_bbox = label['bbox']
        masks = label['mask_target']
        max_id_for_anchors = label['max_id_for_anchors']
        classes = label['classes']
        num_obj = label['num_obj']

        # loc_loss = self._loss_location(pred_offset, box_targets, positiveness) * self._loss_weight_box
        loc_loss = self._loss_giou_location(pred_offset, box_targets, positiveness) * self._loss_weight_box

        # conf_loss = self._loss_class(pred_cls, cls_targets, num_classes, positiveness) * self._loss_weight_cls
        conf_loss = self._focal_conf_objectness_loss(pred_cls, cls_targets, num_classes) * self._loss_weight_cls
        mask_loss = self._loss_mask(proto_out, pred_mask_coef, bbox_norm, masks, positiveness, max_id_for_anchors, max_masks_for_train=100) * self._loss_weight_mask
        seg_loss = self._loss_semantic_segmentation(seg, masks, classes, num_obj) * self._loss_weight_seg
        total_loss = loc_loss + conf_loss + mask_loss + seg_loss

        return loc_loss, conf_loss, mask_loss, seg_loss, total_loss

    def _loss_location(self, pred_offset, gt_offset, positiveness):

        positiveness = tf.expand_dims(positiveness, axis=-1)

        # get postive indices
        pos_indices = tf.where(positiveness == 1)
        pred_offset = tf.gather_nd(pred_offset, pos_indices[:, :-1])
        gt_offset = tf.gather_nd(gt_offset, pos_indices[:, :-1])

        # calculate the smoothL1(positive_pred, positive_gt) and return
        num_pos = tf.shape(gt_offset)[0]
        smoothl1loss = tf.keras.losses.Huber(delta=1., reduction=tf.losses.Reduction.NONE)
        loss_loc = tf.reduce_sum(smoothl1loss(gt_offset, pred_offset)) / tf.cast(num_pos, tf.float32)

        return loss_loc

    def _loss_giou_location(self, pred_offset, gt_offset, positiveness):
        variances = [0.1, 0.2]
        positiveness = tf.expand_dims(positiveness, axis=-1)
        num_batch = tf.shape(pred_offset)[0]

        # pred_offset map to bbox
        anchors = tf.expand_dims(self.anchors.get_anchors(), axis=0)
        anchors = tf.tile(anchors, [num_batch, 1, 1])

        anchors_h = anchors[:, :, 2] - anchors[:, :, 0]
        anchors_w = anchors[:, :, 3] - anchors[:, :, 1]
        anchors_cx = anchors[:, :, 1] + (anchors_w / 2)
        anchors_cy = anchors[:, :, 0] + (anchors_h / 2)

        preds_cx, preds_cy, preds_w, preds_h = tf.unstack(pred_offset, axis=-1)
        
        news_cx = preds_cx * (anchors_w * variances[0]) + anchors_cx
        news_cy = preds_cy * (anchors_h * variances[0]) + anchors_cy
        news_w = tf.math.exp(preds_w * variances[1]) * anchors_w
        news_h = tf.math.exp(preds_h * variances[1]) * anchors_h

        ymins = news_cx - (news_h / 2)
        xmins = news_cx - (news_w / 2)
        ymaxs = news_cy + (news_h / 2)
        xmaxs = news_cx + (news_w / 2)

        pred_bboxes = tf.stack([ymins, xmins, ymaxs, xmaxs], axis=-1)

        # gt_offset map to bbox
        anchors = tf.expand_dims(self.anchors.get_anchors(), axis=0)
        anchors = tf.tile(anchors, [num_batch, 1, 1])

        anchors_h = anchors[:, :, 2] - anchors[:, :, 0]
        anchors_w = anchors[:, :, 3] - anchors[:, :, 1]
        anchors_cx = anchors[:, :, 1] + (anchors_w / 2)
        anchors_cy = anchors[:, :, 0] + (anchors_h / 2)

        gt_cx, gt_cy, gt_w, gt_h = tf.unstack(gt_offset, axis=-1)
        
        news_cx = gt_cx * (anchors_w * variances[0]) + anchors_cx
        news_cy = gt_cy * (anchors_h * variances[0]) + anchors_cy
        news_w = tf.math.exp(gt_w * variances[1]) * anchors_w
        news_h = tf.math.exp(gt_h * variances[1]) * anchors_h

        ymins = news_cx - (news_h / 2)
        xmins = news_cx - (news_w / 2)
        ymaxs = news_cy + (news_h / 2)
        xmaxs = news_cx + (news_w / 2)

        gt_bboxes = tf.stack([ymins, xmins, ymaxs, xmaxs], axis=-1)

        # get positive indices
        pos_indices = tf.where(positiveness == 1)
        pred_bboxes = tf.gather_nd(pred_bboxes, pos_indices[:, :-1])
        gt_bboxes = tf.gather_nd(gt_bboxes, pos_indices[:, :-1])

        # GIoU loss
        num_pos = tf.shape(gt_bboxes)[0]
        loss = tf.reduce_sum(self.giou_loss(gt_bboxes, pred_bboxes)) / tf.cast(num_pos, tf.float32)

        return loss

    def _loss_class(self, pred_cls, gt_cls, num_cls, positiveness):

        # reshape pred_cls from [batch, num_anchor, num_cls] => [batch * num_anchor, num_cls]
        pred_cls = tf.reshape(pred_cls, [-1, num_cls])

        # reshape gt_cls from [batch, num_anchor] => [batch * num_anchor, 1]
        gt_cls = tf.expand_dims(gt_cls, axis=-1)
        gt_cls = tf.reshape(gt_cls, [-1, 1])

        # reshape positiveness to [batch*num_anchor, 1]
        positiveness = tf.expand_dims(positiveness, axis=-1)
        positiveness = tf.reshape(positiveness, [-1, 1])
        pos_indices = tf.where(positiveness == 1)
        neg_indices = tf.where(positiveness == 0)

        # gather pos data, neg data separately
        pos_pred_cls = tf.gather(pred_cls, pos_indices[:, 0])
        pos_gt = tf.gather(gt_cls, pos_indices[:, 0])

        # calculate the needed amount of negative sample
        num_pos = tf.shape(pos_gt)[0]
        num_neg_needed = num_pos * self._neg_pos_ratio

        neg_pred_cls = tf.gather(pred_cls, neg_indices[:, 0])
        neg_gt = tf.gather(gt_cls, neg_indices[:, 0])

        # apply softmax on the pred_cls
        # -log(softmax class 0)
        neg_minus_log_class0 = -1 * tf.nn.log_softmax(neg_pred_cls)[:, 0]

        # sort of -log(softmax class 0)
        neg_minus_log_class0_sort = tf.argsort(neg_minus_log_class0, direction="DESCENDING")

        # take the first num_neg_needed idx in sort result and handle the situation if there are not enough neg
        neg_indices_for_loss = neg_minus_log_class0_sort[:num_neg_needed]

        # combine the indices of pos and neg sample, create the label for them
        neg_pred_cls_for_loss = tf.gather(neg_pred_cls, neg_indices_for_loss)
        neg_gt_for_loss = tf.gather(neg_gt, neg_indices_for_loss)

        # calculate Cross entropy loss and return
        # concat positive and negtive data
        target_logits = tf.concat([pos_pred_cls, neg_pred_cls_for_loss], axis=0)
        target_labels = tf.cast(tf.concat([pos_gt, neg_gt_for_loss], axis=0), tf.int64)
        target_labels = tf.one_hot(tf.squeeze(target_labels), depth=num_cls)

        # total loss
        loss_conf = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(target_labels, target_logits)) / tf.cast(num_pos, tf.float32)

        return loss_conf

    def _focal_loss(self, pred_cls, gt_cls, num_cls):
        pred_cls = tf.reshape(pred_cls, [-1, num_cls]) # [batch * num_anchor, num_cls]
        gt_cls = tf.reshape(gt_cls, [-1]) # [batch * num_anchor]
        return tf.reduce_sum(self.focal_loss_with_logits(y_true=gt_cls, y_pred=pred_cls))

    def _focal_conf_objectness_loss(self, pred_cls, gt_cls, num_cls):
        # Objectness Score First
        pred_cls = tf.reshape(pred_cls, [-1, num_cls])  # [batch * num_anchor, num_cls]
        gt_cls = tf.reshape(gt_cls, [-1])  # [batch * num_anchor]

        keep = tf.cast(gt_cls >= 0, tf.float32)
        gt_cls = tf.nn.relu(gt_cls)  # remove negative value

        foreground = tf.cast(gt_cls != 0, dtype=tf.float32)
        obj_loss = self.focal_loss_with_logits(y_true=foreground, y_pred=pred_cls[:, 0])

        # Now time for the class confidence loss
        pos_mask = gt_cls > 0
        pred_data_positive = (pred_cls[:, 1:])[pos_mask]
        gt_cls_positive = gt_cls[pos_mask] - 1   # Remove background class
        gt_cls_positive = tf.one_hot(tf.cast(gt_cls_positive, tf.int64), depth=num_cls - 1)
        class_loss = tf.nn.softmax_cross_entropy_with_logits(labels=gt_cls_positive, logits=pred_data_positive)

        return tf.reduce_sum(class_loss) / tf.cast( tf.shape(gt_cls_positive)[0], tf.float32 ) + tf.reduce_sum(obj_loss * keep) /  tf.cast( tf.reduce_sum(keep), tf.float32 )

    def _loss_mask(self, proto_output, pred_mask_coef, gt_bbox_norm, gt_masks, positiveness,
                   max_id_for_anchors, max_masks_for_train):

        shape_proto = tf.shape(proto_output)
        num_batch = shape_proto[0]
        loss_mask = 0.
        total_pos = 0
        for idx in tf.range(num_batch):
            # extract randomly postive sample in pred_mask_coef, gt_cls, gt_offset according to positive_indices
            proto = proto_output[idx]
            mask_coef = pred_mask_coef[idx]
            mask_gt = gt_masks[idx]
            bbox_norm = gt_bbox_norm[idx]
            pos = positiveness[idx]
            max_id = max_id_for_anchors[idx]
            pos_indices = tf.squeeze(tf.where(pos == 1))
            # tf.print("num_pos", tf.shape(pos_indices))
            """
            if tf.size(pos_indices) == 0:
                tf.print("detect no positive")
                continue
            """
            # Todo decrease the number pf positive to be 100
            # [num_pos, k]
            pos_mask_coef = tf.gather(mask_coef, pos_indices)
            pos_max_id = tf.gather(max_id, pos_indices)

            if tf.size(pos_indices) == 1:
                # tf.print("detect only one dim")
                pos_mask_coef = tf.expand_dims(pos_mask_coef, axis=0)
                pos_max_id = tf.expand_dims(pos_max_id, axis=0)
            total_pos += tf.size(pos_indices)
            
            # proto = [64, 64, num_mask]
            # pos_mask_coef = [num_pos, num_mask]
            # pred_mask = proto x pos_mask_coef = [64, 64, num_pos]
            # pred_mask transpose = [num_pos, 64, 64]
            pred_mask = tf.linalg.matmul(proto, pos_mask_coef, transpose_a=False, transpose_b=True)
            pred_mask = tf.transpose(pred_mask, perm=(2, 0, 1))

            # calculating loss for each mask coef correspond to each postitive anchor
            # pos_max_id = [num_pos]
            gt = tf.gather(mask_gt, pos_max_id)            # [num_pos, 64, 64]
            bbox = tf.gather(bbox_norm, pos_max_id)        # [num_pos, 4]
            bbox_center = utils.map_to_center_form(bbox)   # [num_pos, 4]
            area = bbox_center[:, -1] * bbox_center[:, -2]

            # crop the pred (not real crop, zero out the area outside the gt box)
            s = tf.nn.sigmoid_cross_entropy_with_logits(gt, pred_mask)  # [num_pos, 64, 64]
            s = utils.crop(s, bbox, origin_w=64, origin_h=64)           # [num_pos, 64, 64]
            loss = tf.reduce_sum(s, axis=[1, 2]) / area                 # [num_pos]
            loss_mask += tf.reduce_sum(loss)

        loss_mask /= tf.cast(total_pos, tf.float32)
        return loss_mask

    def _loss_semantic_segmentation(self, pred_seg, mask_gt, classes, num_obj):

        shape_mask = tf.shape(mask_gt)
        num_batch = shape_mask[0]
        seg_shape = tf.shape(pred_seg)[1]
        loss_seg = 0.

        for idx in tf.range(num_batch):
            seg = pred_seg[idx]
            masks = mask_gt[idx]
            cls = classes[idx]
            objects = num_obj[idx]

            # seg shape (69, 69, num_cls)
            # resize masks from (100, 138, 138) to (100, 69, 69)
            masks = tf.expand_dims(masks, axis=-1)
            masks = tf.image.resize(masks, [seg_shape, seg_shape], method=tf.image.ResizeMethod.BILINEAR)
            masks = tf.cast(masks + 0.5, tf.int64)
            masks = tf.squeeze(tf.cast(masks, tf.float32))

            # obj_mask shape (objects, 138, 138)
            obj_mask = masks[:objects]
            obj_cls = tf.expand_dims(cls[:objects], axis=-1)

            # create empty ground truth (138, 138, num_cls)
            seg_gt = tf.zeros_like(seg)
            seg_gt = tf.transpose(seg_gt, perm=(2, 0, 1))
            seg_gt = tf.tensor_scatter_nd_update(seg_gt, indices=obj_cls, updates=obj_mask)
            seg_gt = tf.transpose(seg_gt, perm=(1, 2, 0))
            loss_seg += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(seg_gt, seg))
        loss_seg = loss_seg / tf.cast(seg_shape, tf.float32) ** 2 / tf.cast(num_batch, tf.float32)

        return loss_seg
