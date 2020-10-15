import tensorflow as tf
import lite
import os
import io
import numpy as np
import PIL
import gridfs

from pymongo import MongoClient
from bson.objectid import ObjectId
from tqdm import tqdm

class TFLiteExporter():
    def __init__(self, model, input_size=256):
        self.model = model
        self.input_shape = (input_size, input_size, 3)
        self.softmax = tf.keras.layers.Softmax()

    def export(self, filename):
        inputs = tf.keras.Input(shape=self.input_shape)
        _, protonet_out, cls_result, offset_result, mask_result = self.model(inputs, training=False)

        wrapper = tf.keras.Model(inputs, [protonet_out, cls_result, offset_result, mask_result])
        converter = tf.lite.TFLiteConverter.from_keras_model(wrapper)
        converter.experimental_new_converter=False
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(filename, 'wb') as F:
            F.write(tflite_model)

class MongoExporter():
    def __init__(self, tfrecord_dir, col_name, subset='train'):
        self.tfrecord_dir = tfrecord_dir
        self.keys2features = {
            'image/source_id': tf.io.FixedLenFeature([], dtype=tf.string),
            'image/height': tf.io.FixedLenFeature([], dtype=tf.int64),
            'image/width': tf.io.FixedLenFeature([], dtype=tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/class/label_id': tf.io.VarLenFeature(dtype=tf.int64),
            'image/object/is_crowd': tf.io.VarLenFeature(dtype=tf.int64),
            'image/object/mask': tf.io.VarLenFeature(dtype=tf.string),
        }

        client = MongoClient(
            'localhost',
            username='mongoadmin',
            password='mongoadmin',
            authSource='admin',
            authMechanism='SCRAM-SHA-256'
        )

        self.collection = client['obj'][col_name]
        self.fs = gridfs.GridFS(client['obj'])

        files = os.listdir(tfrecord_dir)
        files = list(filter(lambda x: subset in x, files))
        files = list(map(lambda x: tfrecord_dir + '/' + x, files))
        dataset = tf.data.TFRecordDataset(files)
        self.dataset = dataset.map(self.parse)
        self.subset = subset
        

    def export(self):
        for data in tqdm(self.dataset):
            masks = self.decode_mask(data)
            img = self.decode_image(data)
            
            height = int(data['image/height'])
            width = int(data['image/width'])
            
            xmin = data['image/object/bbox/xmin'].numpy().tolist()
            xmax = data['image/object/bbox/xmax'].numpy().tolist()
            ymin = data['image/object/bbox/ymin'].numpy().tolist()
            ymax = data['image/object/bbox/ymax'].numpy().tolist()

            label_id = data['image/object/class/label_id'].numpy().tolist()
            is_crowd = data['image/object/is_crowd'].numpy().tolist()
            
            img_io = io.BytesIO()
            img.save(img_io, format='JPEG')
            img_id = self.fs.put(img_io.getvalue())
            
            masks_id = []
            for mask in masks:
                mask_io = io.BytesIO()
                mask.save(mask_io, format='PNG')
                objID = self.fs.put(mask_io.getvalue())
                masks_id.append(objID)

            self.collection.insert_one({
                'image': img_id,
                'masks': masks_id,
                'height': height,
                'width': width,
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax,
                'label_id': label_id,
                'is_crowd': is_crowd,
                'subset': self.subset
            })


    def parse(self, example):
        features = tf.io.parse_single_example(example, self.keys2features)

        for k in features:
            if isinstance(features[k], tf.SparseTensor):
                if features[k].dtype == tf.string:
                    features[k] = tf.sparse.to_dense(features[k], default_value='')
                else:
                    features[k] = tf.sparse.to_dense(features[k], default_value=0)

        return features

    def decode_mask(self, data):
        def decode_png_mask(png_bytes):
            mask = tf.squeeze(
                tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8),
                axis=-1
            )

            mask = tf.cast(mask, dtype=tf.float32)
            mask.set_shape([None, None])

            return mask

        masks = data['image/object/mask']
        masks = tf.cond(
            pred=tf.greater(tf.size(input=masks), 0),
            true_fn=lambda: tf.map_fn(decode_png_mask, masks, dtype=tf.float32),
            false_fn=lambda: tf.zeros([0, height, width], dtype=tf.float32)
        )

        masks = masks.numpy().astype('uint8')
        masks = list(map(PIL.Image.fromarray, masks))
        
        return masks

    def decode_image(self, data):
        image = tf.io.decode_jpeg(data['image/encoded'])
        image.set_shape([None, None, 3])
        image = image.numpy()
        image = PIL.Image.fromarray(image)
        
        return image
    

if __name__ == '__main__':
    exporter = MongoExporter('data/obj_tfrecord_256x256_20200930', 'obj_256x256_20200930', subset='val')
    exporter.export()
    
    exporter = MongoExporter('data/obj_tfrecord_256x256_20200930', 'obj_256x256_20200930', subset='train')
    exporter.export()