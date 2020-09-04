import tensorflow as tf

class ShuffleNetV2():
    def __init__(self, input_shape, dropout=0.1):
        self.input_shape = input_shape

        # First Layer
        self.conv1 = tf.keras.layers.Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.MaxPool2D((3, 3), strides=2, padding='same')
        self.channels = [24, 48, 96, 192]
        self.num_blocks = [4, 8, 4]
        self.stages = [[], [], []]
        self.shorts = []
        
        for stage_id in range(len(self.num_blocks)):
            input_channel = self.channels[stage_id]
            output_channel = self.channels[stage_id + 1]

            self.shorts.append({
                'conv_dw': tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False),
                'bn1': tf.keras.layers.BatchNormalization(axis=-1),
                'conv': tf.keras.layers.Conv2D(input_channel, kernel_size=1, strides=1, padding='same', use_bias=False),
                'bn2': tf.keras.layers.BatchNormalization(axis=-1),
                'relu': tf.keras.layers.ReLU()
            })

            for i in range(self.num_blocks[stage_id]):
                if i == 0:
                    self.stages[stage_id].append({
                        'conv_in': tf.keras.layers.Conv2D(output_channel // 2, kernel_size=1, strides=1, padding='same', use_bias=False),
                        'bn_in': tf.keras.layers.BatchNormalization(axis=-1),
                        'relu_in': tf.keras.layers.ReLU(),
                        'conv_dw': tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False),
                        'conv_bn': tf.keras.layers.BatchNormalization(axis=-1),
                        'conv_out': tf.keras.layers.Conv2D(output_channel - input_channel, kernel_size=1, strides=1, padding='same', use_bias=False),
                        'bn_out': tf.keras.layers.BatchNormalization(axis=-1),
                        'relu_out': tf.keras.layers.ReLU(),
                        'concat': tf.keras.layers.Concatenate(axis=3),
                    })
                else:
                    self.stages[stage_id].append({
                        'conv_in': tf.keras.layers.Conv2D(output_channel // 2, kernel_size=1, strides=1, padding='same', use_bias=False),
                        'bn_in': tf.keras.layers.BatchNormalization(axis=-1),
                        'relu_in': tf.keras.layers.ReLU(),
                        'conv_dw': tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False),
                        'conv_bn': tf.keras.layers.BatchNormalization(axis=-1),
                        'conv_out': tf.keras.layers.Conv2D(output_channel // 2, kernel_size=1, strides=1, padding='same', use_bias=False),
                        'bn_out': tf.keras.layers.BatchNormalization(axis=-1),
                        'relu_out': tf.keras.layers.ReLU(),
                        'concat': tf.keras.layers.Concatenate(axis=3),
                    })

        self.final_conv = tf.keras.layers.Conv2D(1024, kernel_size=1, strides=1, padding='same', use_bias=False)
        self.final_bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.final_relu = tf.keras.layers.ReLU()
            

    def gen(self):
        # First Layer
        inputs = tf.keras.Input(shape=self.input_shape)
        c3 = None
        c4 = None
        c5 = None

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        for stage_id in range(len(self.num_blocks)):
            shortcut, x = tf.split(x, 2, axis=3)
            shortcut = self.shorts[stage_id]['conv_dw'](shortcut)
            shortcut = self.shorts[stage_id]['bn1'](shortcut)
            shortcut = self.shorts[stage_id]['conv'](shortcut)
            shortcut = self.shorts[stage_id]['bn2'](shortcut)
            shortcut = self.shorts[stage_id]['relu'](shortcut)

            x = self.stages[stage_id][0]['conv_in'](x)
            x = self.stages[stage_id][0]['bn_in'](x)
            x = self.stages[stage_id][0]['relu_in'](x)
            x = self.stages[stage_id][0]['conv_dw'](x)
            x = self.stages[stage_id][0]['conv_bn'](x)
            x = self.stages[stage_id][0]['conv_out'](x)
            x = self.stages[stage_id][0]['bn_out'](x)
            x = self.stages[stage_id][0]['relu_out'](x)
            x = self.stages[stage_id][0]['concat']([shortcut, x])
            x = self.channel_shuffle(x)

            for i in range(1, self.num_blocks[stage_id]):
                shortcut, x = tf.split(x, 2, axis=3)
                x = self.stages[stage_id][i]['conv_in'](x)
                x = self.stages[stage_id][i]['bn_in'](x)
                x = self.stages[stage_id][i]['relu_in'](x)
                x = self.stages[stage_id][i]['conv_dw'](x)
                x = self.stages[stage_id][i]['conv_bn'](x)
                x = self.stages[stage_id][i]['conv_out'](x)
                x = self.stages[stage_id][i]['bn_out'](x)
                x = self.stages[stage_id][i]['relu_out'](x)
                x = self.stages[stage_id][i]['concat']([shortcut, x])
                x = self.channel_shuffle(x)

            if stage_id == 0:
                c3 = x
            elif stage_id == 1:
                c4 = x
        
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)

        c5 = x

        model = tf.keras.Model(inputs=inputs, outputs=[c3, c4, c5])
        return model

    def channel_shuffle(self, x, group=2):
        in_shape = x.get_shape().as_list()
        h, w, channel = in_shape[1:]
        x = tf.reshape(x, [-1, h, w, channel // group, group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [-1, h, w, channel])

        return x

if __name__ == '__main__':
    dummy_input = tf.zeros((8, 256, 256, 3))
    model = ShuffleNetV2((256, 256, 3)).gen()
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with tf.io.gfile.GFile('shufflenet_sample.tflite', 'wb') as F:
        F.write(tflite_model)