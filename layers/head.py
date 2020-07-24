import tensorflow as tf

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

        self.input_conv_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.input_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.input_relu_dw = tf.keras.layers.ReLU()
        self.input_conv_pw = tf.keras.layers.Conv2D(
            out_channels, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )
        self.input_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.input_relu_pw = tf.keras.layers.ReLU()
        
        # Class Head
        self.class_conv_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.class_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.class_relu_dw = tf.keras.layers.ReLU()
        self.class_conv_pw = tf.keras.layers.Conv2D(
            self.num_class * self.num_anchors, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )

        self.class_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.class_relu_pw = tf.keras.layers.ReLU()
       
       # Box Head
        self.box_conv_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.box_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.box_relu_dw = tf.keras.layers.ReLU()
        self.box_conv_pw = tf.keras.layers.Conv2D(
            4 * self.num_anchors, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )
        
        self.box_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.box_relu_pw = tf.keras.layers.ReLU()

        # Mask Head
        self.mask_conv_dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            padding='same', 
            depth_multiplier=1, 
            strides=1, 
            use_bias=False
        )

        self.mask_bn_dw = tf.keras.layers.BatchNormalization(axis=-1)
        self.mask_relu_dw = tf.keras.layers.ReLU()
        
        self.mask_conv_pw = tf.keras.layers.Conv2D(
            self.num_mask * self.num_anchors, 
            (1, 1),
            padding='same',
            use_bias=False,
            strides=1,
        )

        self.mask_bn_pw = tf.keras.layers.BatchNormalization(axis=-1)
        self.mask_relu_pw = tf.keras.layers.ReLU()

        self.classReshape = tf.keras.layers.Reshape((-1, num_class))
        self.boxReshape = tf.keras.layers.Reshape((-1, 4))
        self.maskReshape = tf.keras.layers.Reshape((-1, num_mask))

        # Dropout
        self.dropout_input = tf.keras.layers.Dropout(0.1)
        self.dropout_class = tf.keras.layers.Dropout(0.1)
        self.dropout_box = tf.keras.layers.Dropout(0.1)
        self.dropout_mask = tf.keras.layers.Dropout(0.1)

    def call(self, p):
        p = self.input_conv_dw(p)
        p = self.input_bn_dw(p)
        p = self.input_relu_dw(p)
        p = self.input_conv_pw(p)
        p = self.input_bn_pw(p)
        p = self.input_relu_pw(p)
        p = self.dropout_input(p)
        
        # Class Head
        x = self.class_conv_dw(p)
        x = self.class_bn_dw(x)
        x = self.class_relu_dw(x)
        x = self.class_conv_pw(x)
        x = self.class_bn_pw(x)
        pred_class = self.class_relu_pw(x)
        pred_class = self.dropout_class(pred_class)

        # Box Head
        x = self.box_conv_dw(p)
        x = self.box_bn_dw(x)
        x = self.box_relu_dw(x)
        x = self.box_conv_pw(x)
        x = self.box_bn_pw(x)
        pred_box = self.box_relu_pw(x)
        pred_box = self.dropout_box(pred_box)

        # Mask Head
        x = self.mask_conv_dw(p)
        x = self.mask_bn_dw(x)
        x = self.mask_relu_dw(x)
        x = self.mask_conv_pw(x)
        x = self.mask_bn_pw(x)
        pred_mask = self.mask_relu_pw(x)
        pred_mask = self.dropout_mask(pred_mask)

        # reshape the prediction head result for following loss calculation
        pred_class = self.classReshape(pred_class)
        pred_box = self.boxReshape(pred_box)
        pred_mask = self.maskReshape(pred_mask)

        # add activation for conf and mask coef
        pred_mask = tf.keras.activations.tanh(pred_mask)

        return pred_class, pred_box, pred_mask

    