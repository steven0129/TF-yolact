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

class PredictionModule(tf.keras.layers.Layer):

    def __init__(self, out_channels, num_anchors, num_class, num_mask):
        super(PredictionModule, self).__init__()
        self.num_anchors = num_anchors
        self.num_class = num_class
        self.num_mask = num_mask
        self.out_channels = out_channels

        print(f'num_anchors = {self.num_anchors}')
        print(f'num_class = {self.num_class}')
        print(f'num_mask = {self.num_mask}')

        self.input_conv = tf.keras.layers.Conv2D(out_channels, (3, 3), 1, padding='same', kernel_initializer=tf.keras.initializers.glorot_uniform(), activation='relu')
        
        # Class Branch
        self.class_heads = [tf.keras.layers.Conv2D(num_class * num_anchors, (3, 3), 1, padding="same", kernel_initializer=tf.keras.initializers.glorot_uniform())]
        
        # Box Branch
        self.box_heads = [tf.keras.layers.Conv2D(4 * self.num_anchors, (3, 3), 1, padding="same", kernel_initializer=tf.keras.initializers.glorot_uniform())]
        
        # Mask Branch
        self.mask_heads = [tf.keras.layers.Conv2D(self.num_mask * self.num_anchors, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.glorot_uniform())]

        self.classReshape = tf.keras.layers.Reshape((-1, num_class))
        self.boxReshape = tf.keras.layers.Reshape((-1, 4))
        self.maskReshape = tf.keras.layers.Reshape((-1, num_mask))

    def call(self, p):
        p = self.input_conv(p)
        
        pred_class = p
        pred_box = p
        pred_mask = p

        for class_head in self.class_heads:
            pred_class = class_head(pred_class)

        for box_head in self.box_heads:
            pred_box = box_head(pred_box)

        for mask_head in self.mask_heads:
            pred_mask = mask_head(pred_mask)

        # reshape the prediction head result for following loss calculation
        pred_class = self.classReshape(pred_class)
        pred_box = self.boxReshape(pred_box)
        pred_mask = self.maskReshape(pred_mask)

        # add activation for conf and mask coef
        pred_mask = tf.keras.activations.tanh(pred_mask)

        return pred_class, pred_box, pred_mask