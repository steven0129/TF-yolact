import tensorflow as tf

class DwConv(tf.keras.layers.Layer):
    def __init__(self, num_filters, dropout=None):
        super(DwConv, self).__init__()
        self.conv_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.conv_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv_relu_dw = tf.keras.layers.ReLU()
        self.conv_pw = tf.keras.layers.Conv2D(
            num_filters, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )

        self.conv_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv_relu_pw = tf.keras.layers.ReLU()

        if dropout != None:
            self.dropout = tf.keras.layers.Dropout(dropout)
        else:
            self.dropout = None

    def call(self, x):
        x = self.conv_dw(x)
        x = self.conv_bn_dw(x)
        x = self.conv_relu_dw(x)
        x = self.conv_pw(x)
        x = self.conv_bn_pw(x)
        x = self.conv_relu_pw(x)

        if self.dropout != None:
            x = self.dropout(x)

        return x

class ProtoNet(tf.keras.layers.Layer):
    """
        Creating the component of ProtoNet
        Arguments:
            num_prototype
    """

    def __init__(self, num_prototype):
        super(ProtoNet, self).__init__()
        
        self.conv_base = [
            DwConv(96),
            DwConv(96),
            DwConv(96)
        ]

        self.upSampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_up = [DwConv(96)]
        self.conv_final = tf.keras.layers.Conv2D(num_prototype, (3, 3), 1, padding="same", kernel_initializer=tf.keras.initializers.glorot_uniform(), activation="relu")

    def call(self, x):
        for conv in self.conv_base:
            x = conv(x)

        x = self.upSampling(x)
        
        for conv in self.conv_up:
            x = conv(x)

        x = self.conv_final(x)
        
        return x

class PADModule(tf.keras.layers.Layer):
    def __init__(self, num_prototype):
        super(PADModule, self).__init__()
        
        self.p5_conv_1x1 = tf.keras.layers.Conv2D(
            filters=192,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            use_bias=False
        )

        self.p4_conv_3x3 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            padding='same',
            depth_multiplier=1,
            strides=2,
            use_bias=False
        )

        self.p4_conv_1x1 = tf.keras.layers.Conv2D(
            filters=192,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            use_bias=False
        )

        self.p4_sigmoid = tf.keras.layers.Activation('sigmoid')

        self.p3_conv_1x1 = tf.keras.layers.Conv2D(
            filters=192,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            use_bias=False
        )

        self.output_conv_1x1 = tf.keras.layers.Conv2D(
            filters=num_prototype,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            use_bias=False
        )

        self.upSampling4x4 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')
        self.upSampling2x2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

    def call(self, p3, p4, p5):
        # p3 --> 1/8  * input_size
        # p4 --> 1/16 * input_size
        # p5 --> 1/32 * input_size

        p5 = self.p5_conv_1x1(p5)   # [1/32, 1/32, 192]
        p4 = self.p4_conv_3x3(p4)   # [1/32, 1/32, 96]
        p4 = self.p4_conv_1x1(p4)   # [1/32, 1/32, 192]
        p4 = self.p4_sigmoid(p4)    # [1/32, 1/32, 192]
        p3 = self.p3_conv_1x1(p3)   # [1/8, 1/8, 192]

        feature_map = p5 * p4
        feature_map = p5 + feature_map

        feature_map = self.upSampling4x4(feature_map)   # [1/8, 1/8, 192]
        feature_map = p3 + feature_map                  # [1/8, 1/8, 192]
        feature_map = self.upSampling2x2(feature_map)   # [1/4, 1/4, 192]
        prototypes = self.output_conv_1x1(feature_map)  # [1/4, 1/4, 96]

        tf.print(tf.shape(prototypes))

        return prototypes