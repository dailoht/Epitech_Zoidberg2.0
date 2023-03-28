#! /usr/bin/env python3
# coding: utf-8

import tensorflow as tf

def _bytes_feature(value):
    value = tf.io.serialize_tensor(value).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _img_feature(image):
    image = tf.image.encode_jpeg(image).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))


def _image_example(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.convert_image_dtype(image, tf.uint8)
    feature = {
        'image': _img_feature(image),
        'label': _bytes_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def _parse_image_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
    image = tf.ensure_shape(image, (512, 512, 3))
    label = tf.io.parse_tensor(parsed_features['label'], out_type=tf.float32)
    label = tf.ensure_shape(label, (3,))
    return image, label

def save_image_dataset_to_tfrecord(dataset, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for image, label in dataset:
            example = _image_example(image, label)
            writer.write(example.SerializeToString())

def load_image_dataset_from_tfrecord(input_file):
    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(_parse_image_example)
    return dataset

if __name__ == "__main__":
    pass