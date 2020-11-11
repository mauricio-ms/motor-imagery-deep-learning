import os
import re

import numpy as np
import tensorflow as tf


def load(config, filenames, **kwargs):
    path_files = [os.path.join(config["dir"], filename)
                  for filename in filenames]
    return _load_data(path_files, config["window_length"], config["n_channels"], **kwargs)


def get_filenames(config):
    filenames = sorted(os.listdir(config["dir"]))
    filenames = filter(lambda f: re.match(".*.tfrecord", f), filenames)
    return np.array(list(filenames))


def _load_data(path_files, window_length, n_channels,
               n_parallel_calls=tf.data.experimental.AUTOTUNE,
               n_buffer_shuffle=None, batch_size=100, xy_format=False, **kwargs):
    """
        The reason to expand_dim parameter is because TensorFlow expects a certain input shape
        for it's Deep Learning Model. For example a Convolution Neural Network expect:

        (<number of samples>, <x_dim sample>, <y_dim sample>, <number of channels>)
    """
    dataset = tf.data.Dataset.list_files(path_files)
    n_cycle_length = 1
    dataset = dataset.interleave(
        lambda filepath: tf.data.TFRecordDataset(filepath, compression_type="GZIP"),
        cycle_length=n_cycle_length)
    if n_buffer_shuffle is not None:
        dataset = dataset.shuffle(n_buffer_shuffle)
    dataset = dataset.map(lambda r: _preprocess(r, (window_length, n_channels), **kwargs),
                          num_parallel_calls=n_parallel_calls)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if xy_format:
        X = np.empty((0, window_length, n_channels))
        y = np.empty(0, dtype=np.int64)

        for X_batch, y_batch in dataset:
            X = np.vstack((X, X_batch))
            y = np.append(y, y_batch)

        return X, y

    return dataset


@tf.function
def _preprocess(serialized_eeg_records, shape, expand_dim=True):
    """
        The reason to expand_dim parameter is because TensorFlow expects a certain input shape
        for it's Deep Learning Model. For example a Convolution Neural Network expect:

        (<number of samples>, <x_dim sample>, <y_dim sample>, <number of channels>)
    """
    feature_description = {
        "X": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_eeg_records = tf.io.parse_single_example(serialized_eeg_records, feature_description)
    X = parsed_eeg_records["X"]
    X = tf.io.parse_tensor(X, out_type=tf.float64)
    X.set_shape(shape)
    y = parsed_eeg_records["y"]

    if expand_dim:
        X = X[..., np.newaxis]

    return X, y
