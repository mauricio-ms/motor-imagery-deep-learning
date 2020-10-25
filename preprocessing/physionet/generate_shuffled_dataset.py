import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, Feature, BytesList, Int64List

import helpers.file_system_helper as fsh
from preprocessing.physionet.config import PHYSIONET_DIR
from preprocessing.physionet.preprocessing_utils import preprocess


def generate_shuffled_dataset(input_dataset_dir, train_ratio=0.75):
    shuffled_dataset_dir = os.path.join(input_dataset_dir, "shuffled")
    train_dataset_filepath = os.path.join(shuffled_dataset_dir, "train-dataset.tfrecord")
    test_dataset_filepath = os.path.join(shuffled_dataset_dir, "test-dataset.tfrecord")

    input_dataset_dir_parts = input_dataset_dir.split(os.sep)
    try:
        shape_part = next(filter(lambda p: re.match("^(\\d+|(\\d+)?x)+$", p), input_dataset_dir_parts))
        X_shape = [int(dimension) for dimension in shape_part.split("x")]
    except Exception:
        raise AttributeError("Unexpected value for parameter 'shuffled_dataset_dir'.")
    if len(X_shape) == 0:
        raise AttributeError("Unexpected value for parameter 'shuffled_dataset_dir'.")

    fsh.recreate_dir(shuffled_dataset_dir)

    print("Reading dataset ...")
    path_files = [os.path.join(input_dataset_dir, filename)
                  for filename in sorted(os.listdir(input_dataset_dir))
                  if re.match("S(\\d+).tfrecord", filename)]
    dataset = tf.data.Dataset.list_files(path_files)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TFRecordDataset(filepath, compression_type="GZIP"))
    dataset = dataset.map(lambda r: preprocess(r, X_shape=X_shape))
    dataset = dataset.batch(20000)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    X_data = np.empty((0, *X_shape))
    y_data = np.empty(0, dtype=np.int64)
    for X, y in dataset:
        X_data = np.vstack((X_data, X.numpy()))
        y_data = np.append(y_data, y.numpy())

    n_samples = len(y_data)
    y_data = y_data.reshape(-1)

    print("Shuffling dataset ...")
    shuffled_indexes = np.arange(n_samples)
    np.random.shuffle(shuffled_indexes)
    X_data = X_data[shuffled_indexes]
    y_data = y_data[shuffled_indexes]

    train_samples_selection = np.random.rand(n_samples) < train_ratio

    X_train = X_data[train_samples_selection]
    y_train = y_data[train_samples_selection]
    X_test = X_data[~train_samples_selection]
    y_test = y_data[~train_samples_selection]

    print("Writing train TFRecords ...")
    print("Train samples=", len(y_train))
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(train_dataset_filepath, options) as train_writer:
        for X, y in zip(X_train, y_train):
            eeg_example = Example(
                features=Features(
                    feature={
                        "X": Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(X).numpy()])),
                        "y": Feature(int64_list=Int64List(value=[y]))
                    }
                )
            )
            train_writer.write(eeg_example.SerializeToString())

    print("Writing test TFRecords ...")
    print("Test samples=", len(y_test))
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(test_dataset_filepath, options) as test_writer:
        for X, y in zip(X_test, y_test):
            eeg_example = Example(
                features=Features(
                    feature={
                        "X": Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(X).numpy()])),
                        "y": Feature(int64_list=Int64List(value=[y]))
                    }
                )
            )
            test_writer.write(eeg_example.SerializeToString())


generate_shuffled_dataset(os.path.join(PHYSIONET_DIR, "normalized-by-sample",
                                       "window-10-offset-5", "10x64",
                                       "eyes-closed-left-fist-right-fist-both-fists-both-feet"))
