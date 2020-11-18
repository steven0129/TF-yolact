"""
YOLACT:Real-time Instance Segmentation
Ref: https://arxiv.org/abs/1904.02689

Arthor: HSU, CHIHCHAO
"""
import tensorflow as tf

from layers.fpn import FeaturePyramidNeck
from layers.head import PredictionModule
from layers.protonet import PADModule, ProtoNet
from layers.pretrained import MobileNetV1
from layers.pretrained import MobileNetV2
from layers.shufflenet import ShuffleNetV2
from utils.create_prior import make_priors

assert tf.__version__.startswith('2')

class MyYolact():
    def __init__(self, input_size, fpn_channels, feature_map_size, num_class, num_mask, aspect_ratio, scales):
        # use pre-trained MobileNetV2
        self.input_shape = (input_size, input_size, 3)
        self.backbone_pretrained = MobileNetV2(input_shape=(self.input_shape)).gen()
        self.backbone_pretrained.trainable = True
        self.use_padmodule = False

        # extract certain feature maps for FPN
        self.backbone_fpn = FeaturePyramidNeck(fpn_channels, bidirectional=True)
        
        if self.use_padmodule:
            self.protonet = PADModule(num_mask)
        else:
            self.protonet = ProtoNet(num_mask)

        # semantic segmentation branch to boost feature richness
        self.semantic_segmentation = tf.keras.layers.Conv2D(num_class, (1, 1), 1, padding="same",
                                                            kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.num_anchor, self.priors = make_priors(input_size, feature_map_size, aspect_ratio, scales)
        print("prior shape:", self.priors.shape)
        print("num anchor per feature map: ", self.num_anchor)

        # shared prediction head
        self.pred_head = [
            PredictionModule(fpn_channels, len(aspect_ratio), num_class, num_mask),
            PredictionModule(fpn_channels, len(aspect_ratio), num_class, num_mask),
            PredictionModule(fpn_channels, len(aspect_ratio), num_class, num_mask),
            PredictionModule(fpn_channels, len(aspect_ratio), num_class, num_mask),
            PredictionModule(fpn_channels, len(aspect_ratio), num_class, num_mask)
        ]

        self.concat = tf.keras.layers.Concatenate(axis=1)


    def gen(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        c3, c4, c5, c6, c7 = self.backbone_pretrained(inputs)
        fpn_out = self.backbone_fpn(c3, c4, c5, c6, c7)
        p3, p4, p5, _, _ = fpn_out

        # Protonet Branch
        if self.use_padmodule:
            protonet_out = self.protonet(p3, p4, p5)
        else:
            protonet_out = self.protonet(p3)

        # Semantic Segmentation Branch
        seg = self.semantic_segmentation(p3)

        # Prediction Head Branch
        all_pred_cls = []
        all_pred_offset = []
        all_pred_mask = []

        for idx, f_map in enumerate(fpn_out):
            pred_cls, pred_offset, pred_mask = self.pred_head[idx](f_map)
            all_pred_cls.append(pred_cls)
            all_pred_offset.append(pred_offset)
            all_pred_mask.append(pred_mask)

        cls_result = self.concat(all_pred_cls)
        offset_result = self.concat(all_pred_offset)
        mask_result = self.concat(all_pred_mask)

        model = tf.keras.Model(inputs, [seg, protonet_out, cls_result, offset_result, mask_result])

        return model

class TFLiteExporter():
    def __init__(self, model, input_size=256):
        self.model = model
        self.input_shape = (input_size, input_size, 3)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.softmax = tf.keras.layers.Softmax()

    def export(self, filename):
        inputs = tf.keras.Input(shape=self.input_shape)
        _, protonet_out, cls_result, offset_result, mask_result = self.model(inputs)
        cls_result = self.softmax(cls_result)

        wrapper = tf.keras.Model(inputs, [protonet_out, cls_result, offset_result, mask_result])
        converter = tf.lite.TFLiteConverter.from_keras_model(wrapper)
        converter.experimental_new_converter=False
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(filename, 'wb') as F:
            F.write(tflite_model)

if __name__ == '__main__':
    YOLACT = MyYolact(
        input_size=256,
        fpn_channels=96, 
        feature_map_size=[32, 16, 8, 4, 2],
        num_class=13, # 12 classes + 1 background
        num_mask=32,
        aspect_ratio=[1, 0.5, 2],
        scales=[24, 48, 96, 192, 384]
    )

    model = YOLACT.gen()
    exporter = TFLiteExporter(model)
    exporter.export('yolact.tflite')
