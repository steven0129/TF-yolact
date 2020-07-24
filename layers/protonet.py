import tensorflow as tf


class ProtoNet(tf.keras.layers.Layer):
    """
        Creating the component of ProtoNet
        Arguments:
            num_prototype
    """

    def __init__(self, num_prototype):
        super(ProtoNet, self).__init__()
        # Conv1 Layer
        self.conv1_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.conv1_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv1_relu_dw = tf.keras.layers.ReLU()
        self.conv1_pw = tf.keras.layers.Conv2D(
            160, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )
        self.conv1_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv1_relu_pw = tf.keras.layers.ReLU()

        # Conv2 Layer
        self.conv2_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.conv2_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv2_relu_dw = tf.keras.layers.ReLU()
        self.conv2_pw = tf.keras.layers.Conv2D(
            160, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )
        self.conv2_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv2_relu_pw = tf.keras.layers.ReLU()



        # Conv3 Layer
        self.conv3_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.conv3_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv3_relu_dw = tf.keras.layers.ReLU()
        self.conv3_pw = tf.keras.layers.Conv2D(
            160, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )

        self.conv3_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv3_relu_pw = tf.keras.layers.ReLU()

        self.upSampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        # Conv4 Layer
        self.conv4_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.conv4_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv4_relu_dw = tf.keras.layers.ReLU()
        self.conv4_pw = tf.keras.layers.Conv2D(
            160, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )
        self.conv4_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv4_relu_pw = tf.keras.layers.ReLU()

        # Final Conv Layer
        self.final_conv_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.final_conv_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.final_conv_relu_dw = tf.keras.layers.ReLU()
        self.final_conv_pw = tf.keras.layers.Conv2D(
            num_prototype, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )
        self.final_conv_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.final_conv_relu_pw = tf.keras.layers.ReLU()

        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.dropout3 = tf.keras.layers.Dropout(0.1)
        self.dropout4 = tf.keras.layers.Dropout(0.1)
        self.dropout_final = tf.keras.layers.Dropout(0.1)

    def call(self, x):
        x = self.conv1_dw(x)
        x = self.conv1_bn_dw(x)
        x = self.conv1_relu_dw(x)
        x = self.conv1_pw(x)
        x = self.conv1_bn_pw(x)
        x = self.conv1_relu_pw(x)
        x = self.dropout1(x)

        x = self.conv2_dw(x)
        x = self.conv2_bn_dw(x)
        x = self.conv2_relu_dw(x)
        x = self.conv2_pw(x)
        x = self.conv2_bn_pw(x)
        x = self.conv2_relu_pw(x)
        x = self.dropout2(x)

        x = self.conv3_dw(x)
        x = self.conv3_bn_dw(x)
        x = self.conv3_relu_dw(x)
        x = self.conv3_pw(x)
        x = self.conv3_bn_pw(x)
        x = self.conv3_relu_pw(x)
        x = self.dropout3(x)

        # upsampling + convolution
        x = self.upSampling(x)
        
        x = self.conv4_dw(x)
        x = self.conv4_bn_dw(x)
        x = self.conv4_relu_dw(x)
        x = self.conv4_pw(x)
        x = self.conv4_bn_pw(x)
        x = self.conv4_relu_pw(x)
        x = self.dropout4(x)

        # final convolution
        x = self.final_conv_dw(x)
        x = self.final_conv_bn_dw(x)
        x = self.final_conv_relu_dw(x)
        x = self.final_conv_pw(x)
        x = self.final_conv_bn_pw(x)
        x = self.final_conv_relu_pw(x)
        x = self.dropout_final(x)
        
        return x
