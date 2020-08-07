import tensorflow as tf

class MobileNetV2():
    def __init__(self, input_shape, dropout=0.1):
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, 
            include_top=False, 
            layers=tf.keras.layers, 
            weights='imagenet'
        )

        self.input_shape = input_shape
        self.base_model.trainable = True
        self.conv1_pad = self.base_model.get_layer('Conv1_pad')
        self.conv1 = self.base_model.get_layer('Conv1')
        self.conv1_bn = self.base_model.get_layer('bn_Conv1')
        self.conv1_relu = tf.keras.layers.ReLU()
        self.dropout1 = tf.keras.layers.Dropout(dropout)

        self.conv_dw_expand = self.base_model.get_layer('expanded_conv_depthwise')
        self.conv_dw_bn_expand = self.base_model.get_layer('expanded_conv_depthwise_BN')
        self.conv_dw_relu_expand = tf.keras.layers.ReLU()
        self.conv_project_expand = self.base_model.get_layer('expanded_conv_project')
        self.conv_project_bn_expand = self.base_model.get_layer('expanded_conv_project_BN')
        self.dropout2 = tf.keras.layers.Dropout(dropout)

        self.blocks = []

        for i in range(1, 17):
            block = {
                'expand': self.base_model.get_layer(f'block_{i}_expand'),
                'expand_bn': self.base_model.get_layer(f'block_{i}_expand_BN'),
                'expand_relu': tf.keras.layers.ReLU(),
                'conv_dw': self.base_model.get_layer(f'block_{i}_depthwise'),
                'conv_dw_bn': self.base_model.get_layer(f'block_{i}_depthwise_BN'),
                'conv_dw_relu': tf.keras.layers.ReLU(),
                'project': self.base_model.get_layer(f'block_{i}_project'),
                'project_bn': self.base_model.get_layer(f'block_{i}_project_BN'),
                'dropout': tf.keras.layers.Dropout(dropout)
            }

            if i in [1, 3, 6, 13]:
                block['padding'] = self.base_model.get_layer(f'block_{i}_pad')

            if i in [2, 4, 5, 7, 8, 9, 11, 12, 14, 15]:
                block['add'] = self.base_model.get_layer(f'block_{i}_add')

            self.blocks.append(block)
        
        self.final_conv = self.base_model.layers[-3]
        self.final_conv_bn = self.base_model.layers[-2]
        self.final_conv_relu = tf.keras.layers.ReLU()

    def gen(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        c3 = None
        c4 = None
        c5 = None

        x = self.conv1_pad(inputs)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        x = self.dropout1(x)

        x = self.conv_dw_expand(x)
        x = self.conv_dw_bn_expand(x)
        x = self.conv_dw_relu_expand(x)
        x = self.conv_project_expand(x)
        x = self.conv_project_bn_expand(x)
        x = self.dropout2(x)

        for idx, block in enumerate(self.blocks):
            block_in = x
            x = block['expand'](x)
            x = block['expand_bn'](x)
            x = block['expand_relu'](x)

            if 'padding' in block:
                x = block['padding'](x)

            x = block['conv_dw'](x)
            x = block['conv_dw_bn'](x)
            x = block['conv_dw_relu'](x)

            x = block['project'](x)
            x = block['project_bn'](x)
            x = block['dropout'](x)

            if 'add' in block:
                x = block['add']([x, block_in])

            if idx + 1 == 5:
                c3 = x

            if idx + 1 == 9:
                c4 = x

            if idx + 1 == 14:
                c5 = x
        
        x = self.final_conv(x)
        x = self.final_conv_bn(x)
        x = self.final_conv_relu(x)

        model = tf.keras.Model(inputs=inputs, outputs=[c3, c4, c5])
        return model


if __name__ == '__main__':
    dummy_input = tf.zeros((8, 320, 320, 3))
    model = MobileNetV2(input_shape=(320, 320, 3)).gen()
    model.summary()
    out = ['block_5_add', 'block_9_add', 'block_14_add']

    y = model(dummy_input, training=False)
    tf.print(tf.shape(y[0]))
    tf.print(tf.shape(y[1]))
    tf.print(tf.shape(y[2]))