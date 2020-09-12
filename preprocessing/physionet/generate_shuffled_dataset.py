import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, Feature, BytesList, Int64List

import helpers.file_system_helper as fsh
from preprocessing.physionet.config import PHYSIONET_DIR, EXECUTIONS_OF_INTEREST
from preprocessing.physionet.preprocessing_utils import preprocess_3d

INPUT_DATASET_DIR = os.path.join(PHYSIONET_DIR, "3D", "tfrecord-files",
                                 "normalized-by-sample", "-".join(EXECUTIONS_OF_INTEREST))
TRAIN_DATASET_FILEPATH = os.path.join(INPUT_DATASET_DIR, "train-dataset.tfrecord")
TEST_DATASET_FILEPATH = os.path.join(INPUT_DATASET_DIR, "test-dataset.tfrecord")

fsh.rm_if_exists(TRAIN_DATASET_FILEPATH)
fsh.rm_if_exists(TEST_DATASET_FILEPATH)

print("Reading dataset ...")
path_files = [os.path.join(INPUT_DATASET_DIR, filename)
              for filename in sorted(os.listdir(INPUT_DATASET_DIR))
              if re.match("S(\\d+).tfrecord", filename)]
dataset = tf.data.Dataset.list_files(path_files)
dataset = dataset.interleave(
    lambda filepath: tf.data.TFRecordDataset(filepath, compression_type="GZIP"))
dataset = dataset.map(preprocess_3d)
dataset = dataset.batch(20000)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

X_data = np.empty((0, 10, 10, 11))
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

train_ratio = 0.75
train_samples_selection = np.random.rand(n_samples) < train_ratio

X_train = X_data[train_samples_selection]
y_train = y_data[train_samples_selection]
X_test = X_data[~train_samples_selection]
y_test = y_data[~train_samples_selection]

print("Writing train TFRecords ...")
print("Train samples=", len(y_train))
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(TRAIN_DATASET_FILEPATH, options) as train_writer:
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
with tf.io.TFRecordWriter(TEST_DATASET_FILEPATH, options) as test_writer:
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
