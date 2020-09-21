import tensorflow as tf
import lite

class TFLiteExporter():
    def __init__(self, model, input_size=256):
        self.model = model
        self.input_shape = (input_size, input_size, 3)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.softmax = tf.keras.layers.Softmax()

    def export(self, filename):
        inputs = tf.keras.Input(shape=self.input_shape)
        _, protonet_out, cls_result, offset_result, mask_result = model(inputs)
        cls_result = self.softmax(cls_result)
        mask_result = self.sigmoid(mask_result)

        wrapper = tf.keras.Model(inputs, [protonet_out, cls_result, offset_result, mask_result])
        converter = tf.lite.TFLiteConverter.from_keras_model(wrapper)
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(filename, 'wb') as F:
            F.write(tflite_model)

if __name__ == '__main__':
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

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

    exporter = TFLiteExporter(model)
    exporter.export('yolact-20200921.tflite')