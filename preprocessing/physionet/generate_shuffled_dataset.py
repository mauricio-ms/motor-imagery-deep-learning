import os

import numpy as np
import tensorflow as tf

import helpers.file_system_helper as fsh
from preprocessing.physionet.config import PHYSIONET_DIR, EXECUTIONS_OF_INTEREST

INPUT_DATASET_DIR = os.path.join(PHYSIONET_DIR, "3D", "tfrecord-files",
                                 "normalized-by-sample", "-".join(EXECUTIONS_OF_INTEREST))
TRAIN_DATASET_FILEPATH = os.path.join(INPUT_DATASET_DIR, "train-dataset.tfrecord")
TEST_DATASET_FILEPATH = os.path.join(INPUT_DATASET_DIR, "test-dataset.tfrecord")

fsh.rm_if_exists(TRAIN_DATASET_FILEPATH)
fsh.rm_if_exists(TEST_DATASET_FILEPATH)

print("Reading dataset ...")
path_files = [os.path.join(INPUT_DATASET_DIR, subject)
              for subject in sorted(os.listdir(INPUT_DATASET_DIR))]
dataset = tf.data.Dataset.list_files(path_files)
dataset = dataset.interleave(
    lambda filepath: tf.data.TFRecordDataset(filepath, compression_type="GZIP"))

samples = np.array(list(dataset.as_numpy_iterator()))
n_samples = len(samples)

shuffled_indexes = np.arange(n_samples)
np.random.shuffle(shuffled_indexes)
samples = samples[shuffled_indexes]

train_ratio = 0.75
train_samples_selection = np.random.rand(n_samples) < train_ratio

samples_train = samples[train_samples_selection]
samples_test = samples[~train_samples_selection]

print("Writing train TFRecords ...")
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(TRAIN_DATASET_FILEPATH, options) as train_writer:
    for sample_train in samples_train:
        train_writer.write(sample_train)

print("Writing test TFRecords ...")
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(TEST_DATASET_FILEPATH, options) as test_writer:
    for sample_test in samples_test:
        test_writer.write(sample_test)
