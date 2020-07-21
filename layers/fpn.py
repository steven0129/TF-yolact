import tensorflow as tf


class FeaturePyramidNeck(tf.keras.layers.Layer):
    """
        Creating the backbone component of feature Pyramid Network
        Arguments:
            num_fpn_filters
    """

    def __init__(self, num_fpn_filters):
        super(FeaturePyramidNeck, self).__init__()
        self.upSample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        # no Relu for downsample layer
        self.downSample1 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.downSample2 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.lateralCov1 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralCov2 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralCov3 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())

        # Predict Layer P5
        self.predictP5_conv_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.predictP5_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.predictP5_relu_dw = tf.keras.layers.ReLU()
        self.predictP5_conv_pw = tf.keras.layers.Conv2D(
            num_fpn_filters, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )

        self.predictP5_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.predictP5_relu_pw = tf.keras.layers.ReLU()

        # Predict Layer P4
        self.predictP4_conv_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.predictP4_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.predictP4_relu_dw = tf.keras.layers.ReLU()
        self.predictP4_conv_pw = tf.keras.layers.Conv2D(
            num_fpn_filters, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )

        self.predictP4_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.predictP4_relu_pw = tf.keras.layers.ReLU()


        # Predict Layer P3
        self.predictP3_conv_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.predictP3_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.predictP3_relu_dw = tf.keras.layers.ReLU()
        self.predictP3_conv_pw = tf.keras.layers.Conv2D(
            num_fpn_filters, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )

        self.predictP3_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.predictP3_relu_pw = tf.keras.layers.ReLU()


    def call(self, c3, c4, c5):
        # lateral conv for c3 c4 c5
        p5 = self.lateralCov1(c5)
        p4 = self._crop_and_add(self.upSample(p5), self.lateralCov2(c4))
        p3 = self._crop_and_add(self.upSample(p4), self.lateralCov3(c3))
        # print("p3: ", p3.shape)

        # smooth pred layer for p3, p4, p5
        p3 = self.predictP3_conv_dw(p3)
        p3 = self.predictP3_bn_dw(p3)
        p3 = self.predictP3_relu_dw(p3)
        p3 = self.predictP3_conv_pw(p3)
        p3 = self.predictP3_bn_pw(p3)
        p3 = self.predictP3_relu_pw(p3)
        
        p4 = self.predictP4_conv_dw(p4)
        p4 = self.predictP4_bn_dw(p4)
        p4 = self.predictP4_relu_dw(p4)
        p4 = self.predictP4_conv_pw(p4)
        p4 = self.predictP4_bn_pw(p4)
        p4 = self.predictP4_relu_pw(p4)
        
        p5 = self.predictP5_conv_dw(p5)
        p5 = self.predictP5_bn_dw(p5)
        p5 = self.predictP5_relu_dw(p5)
        p5 = self.predictP5_conv_pw(p5)
        p5 = self.predictP5_bn_pw(p5)
        p5 = self.predictP5_relu_pw(p5)

        # downsample conv to get p6, p7
        p6 = self.downSample1(p5)
        p7 = self.downSample2(p6)

        return [p3, p4, p5, p6, p7]

    def _crop_and_add(self, x1, x2):
        """
        for p4, c4; p3, c3 to concatenate with matched shape
        https://tf-unet.readthedocs.io/en/latest/_modules/tf_unet/layers.html
        """
        x1_shape = x1.shape
        x2_shape = x2.shape
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.add(x1_crop, x2)
