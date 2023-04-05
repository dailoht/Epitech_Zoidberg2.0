"""
This module provides functions for working with image datasets in TensorFlow.
It includes functions for saving and loading image datasets to/from TFRecord
format, defining a distribution strategy for training on multiple devices, and
parsing example protos to extract images and labels.
"""

import tensorflow as tf


def _bytes_feature(value):
    """
    Returns a `Feature` object with bytes_list containing the serialized
    input tensor.

    Args:
        value (tf.Tensor): Input tensor to be serialized.

    Returns:
        tf.train.Feature: Serialized tensor wrapped in a `Feature` object.
    """
    value = tf.io.serialize_tensor(value).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _img_feature(image):
    """
    Returns a `Feature` object with bytes_list containing the encoded JPEG
    image.

    Args:
        image (tf.Tensor): Input image to be encoded.

    Returns:
        tf.train.Feature: Encoded image wrapped in a `Feature` object.
    """
    image = tf.image.encode_jpeg(image, quality=100).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))


def _image_example(image, label):
    """
    Returns an `Example` object with the input image and label as features.

    Args:
        image (tf.Tensor): Input image to be wrapped in a `Feature`.
        label (tf.Tensor): Input label to be wrapped in a `Feature`.

    Returns:
        tf.train.Example: `Example` object with the input image and label as
            features.
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.convert_image_dtype(image, tf.uint8)
    feature = {
        'image': _img_feature(image),
        'label': _bytes_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _parse_image_example(example_proto):
    """
    Parses a single example (image and label) from a serialized string
    Tensor.

    Args:
        example_proto (tf.Tensor): The serialized image example.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the decoded image and
        its label.
    """
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example_proto,
                                                 feature_description
                                                 )
    image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
    image = tf.ensure_shape(image, (512, 512, 3))
    label = tf.io.parse_tensor(parsed_features['label'], out_type=tf.float32)
    label = tf.ensure_shape(label, (3,))
    return image, label


def save_image_dataset_to_tfrecord(dataset, output_file):
    """
    Save a dataset of images to a TFRecord file.

    Args:
        dataset (tf.data.Dataset): The dataset to be saved.
        output_file (str): The path to the output file.

    Returns:
        None
    """
    with tf.io.TFRecordWriter(output_file) as writer:
        for image, label in dataset:
            example = _image_example(image, label)
            writer.write(example.SerializeToString())


def load_image_dataset_from_tfrecord(input_file,
                                     prefetch=False,
                                     shuffle_size=1024,
                                     batch_size=32,
                                     opt=None):
    """
    Load a dataset of images from a TFRecord file.

    Args:
        input_file (str): The path to the input file.
        prefetch (bool, optional): If True, prefetch the dataset. Defaults to
            False.
        shuffle_size (int, optional): The buffer size used for shuffling.
            Defaults to 1024.
        batch_size (int, optional): The batch size. Defaults to 32.
        opt (Optional[tf.data.Options], optional): Optional dataset options.
            Defaults to None.

    Returns:
        tf.data.Dataset: The loaded dataset.
    """
    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(_parse_image_example)
    if prefetch:
        dataset = dataset.shuffle(buffer_size=shuffle_size, seed=42)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    if opt:
        dataset = dataset.with_options(opt)
    return dataset


def _tpu_strategy():
    """
    This function attempts to connect to a TPU cluster using a
    TPUClusterResolver, and initializes the TPU system.

    Returns:
        tf.distribute.TPUStrategy or None: A distributed training strategy for
            TPU devices, or None if TPU devices are not available.
    """
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except ValueError:
        strategy = None
    return strategy


def _gpu_strategy():
    """
    This function checks if any GPU devices are available using
    tf.config.list_physical_devices().

    Returns:
        tf.distribute.MirroredStrategy or None: A distributed training
            strategy for GPU devices, or None if GPU devices are not
            available.
    """
    try:
        if not tf.config.list_physical_devices('GPU'):
            raise ValueError("GPU devices not found")
        strategy = tf.distribute.MirroredStrategy()
    except ValueError:
        strategy = None
    return strategy


def _cpu_strategy():
    """
    This function checks if any CPU devices are available using
    tf.config.list_physical_devices().

    Returns:
        tf.distribute.Strategy or None: A distributed training strategy for
            CPU devices, or None if CPU devices are not available.
    """
    try:
        if not tf.config.list_physical_devices('CPU'):
            raise ValueError("CPU devices not found")
        strategy = tf.distribute.get_strategy()
    except ValueError:
        strategy = None
    return strategy


def define_distribute_strategy(device_type='AUTO'):
    """
    Defines and returns a TensorFlow distributed training strategy.

    Args:
        device_type (str): The type of device to use for distributed training.
            Can be "AUTO", "TPU", "GPU", or "CPU". Default is "AUTO".

    Raises:
        ValueError: If the specified device type is not found or if a
            distribution strategy cannot be created.

    Returns:
        tf.distribute.Strategy: A distributed training strategy object for the
            specified device type.
    """
    strategies = {
        'TPU': _tpu_strategy,
        'GPU': _gpu_strategy,
        'CPU': _cpu_strategy,
    }

    if device_type == 'AUTO':
        for strategy_func in strategies.values():
            strategy = strategy_func()
            if strategy:
                print(f"Selected distribution strategy: \
                    {strategy.__class__.__name__}")
                return strategy
        raise ValueError("Unable to find a distribution strategy")

    if device_type not in strategies:
        raise ValueError("Unable to find a distribution strategy \
            for specified device type")

    strategy = strategies[device_type]()
    if strategy is None:
        raise ValueError("Unable to find a distribution strategy \
            for specified device type")
    else:
        print(f"Selected distribution strategy: {strategy.__class__.__name__}")
    return strategy


if __name__ == "__main__":
    pass
