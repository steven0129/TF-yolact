"""
YOLACT:Real-time Instance Segmentation
Ref: https://arxiv.org/abs/1904.02689

Arthor: HSU, CHIHCHAO
"""
import tensorflow as tf

from layers.fpn import FeaturePyramidNeck
from layers.head import PredictionModule
from layers.protonet import ProtoNet
from layers.pretrained import MobileNetV2
from utils.create_prior import make_priors

assert tf.__version__.startswith('2')

class MyYolact():
    def __init__(self, input_size, fpn_channels, feature_map_size, num_class, num_mask, aspect_ratio, scales):
        # use pre-trained MobileNetV2
        self.input_shape = (input_size, input_size, 3)
        self.backbone_pretrained = MobileNetV2(input_shape=(self.input_shape)).gen()
        self.backbone_pretrained.trainable = True

        # extract certain feature maps for FPN
        self.backbone_fpn = FeaturePyramidNeck(fpn_channels)
        self.protonet = ProtoNet(num_mask)

        # semantic segmentation branch to boost feature richness
        self.semantic_segmentation = tf.keras.layers.Conv2D(num_class, (1, 1), 1, padding="same",
                                                            kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.num_anchor, self.priors = make_priors(input_size, feature_map_size, aspect_ratio, scales)
        print("prior shape:", self.priors.shape)
        print("num anchor per feature map: ", self.num_anchor)

        # shared prediction head
        self.pred_head = PredictionModule(fpn_channels, len(aspect_ratio), num_class, num_mask)
        self.concat = tf.keras.layers.Concatenate(axis=1)


    def gen(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        c2, c3, c4, c5 = self.backbone_pretrained(inputs)
        fpn_out = self.backbone_fpn(c2, c3, c4, c5)
        p2 = fpn_out[0]

        # Protonet Branch
        protonet_out = self.protonet(p2)

        # Semantic Segmentation Branch
        seg = self.semantic_segmentation(p2)

        # Prediction Head Branch
        all_pred_cls = []
        all_pred_offset = []
        all_pred_mask = []

        for f_map in fpn_out:
            pred_cls, pred_offset, pred_mask = self.pred_head(f_map)
            all_pred_cls.append(pred_cls)
            all_pred_offset.append(pred_offset)
            all_pred_mask.append(pred_mask)

        cls_result = self.concat(all_pred_cls)
        offset_result = self.concat(all_pred_offset)
        mask_result = self.concat(all_pred_mask)

        model = tf.keras.Model(inputs, [seg, protonet_out, cls_result, offset_result, mask_result])

        return model

if __name__ == '__main__':
    YOLACT = MyYolact(
        input_size=320,
        fpn_channels=256, 
        feature_map_size=[80, 40, 20, 10, 5, 3],
        num_class=13, # 12 classes + 1 background
        num_mask=32,
        aspect_ratio=[1, 0.5, 2],
        scales=[12, 24, 48, 96, 192, 384]
    )

    model = YOLACT.gen()
    model.save('yolact.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with tf.io.gfile.GFile('yolact.tflite', 'wb') as F:
        F.write(tflite_model)
