import os
from main import ROOT_DIR
from logger.log_factory import get_logger
import tensorflow as tf
import numpy as np

LOGGER = get_logger("physionet.py")
PHYSIONET_DIR = os.path.join(ROOT_DIR, "data/physionet")
TFRECORD_FILES_DIR = os.path.join(PHYSIONET_DIR, "preprocessed-tfrecord-files")


def load_data(train_size=0.75, validation_size=None, n_subjects=None,
              train_subjects=None, test_subjects=None, **kwargs):
    LOGGER.info("Loading Physionet dataset ...")
    subjects = np.array(sorted(os.listdir(TFRECORD_FILES_DIR)))
    if n_subjects is not None:
        np.random.shuffle(subjects)
        subjects = subjects[:n_subjects]

    if train_subjects is None or test_subjects is None:
        train_subjects, test_subjects = _train_test_split_subjects(subjects, train_size)

    if validation_size is not None:
        train_subjects, validation_subjects = _train_test_split_subjects(train_subjects, 1-validation_size)
        LOGGER.info(f"(Train, Validation, Test) Subjects = "
                    f"({len(train_subjects)}, {len(validation_subjects)}, {len(test_subjects)})")
        LOGGER.info(f"Train subjects: {train_subjects}")
        LOGGER.info(f"Validation subjects: {validation_subjects}")
        LOGGER.info(f"Test subjects: {test_subjects}")
        return _load_set(train_subjects, **kwargs), \
            _load_set(validation_subjects, **kwargs), \
            _load_set(test_subjects, **kwargs)

    LOGGER.info(f"(Train, Test) Subjects = ({len(train_subjects)}, {len(test_subjects)})")
    LOGGER.info(f"Train subjects: {train_subjects}")
    LOGGER.info(f"Test subjects: {test_subjects}")
    return _load_set(train_subjects, **kwargs), _load_set(test_subjects, **kwargs)


def _train_test_split_subjects(subjects, train_size):
    train_subjects_mask = np.random.rand(len(subjects)) < train_size
    return subjects[train_subjects_mask], subjects[~train_subjects_mask]


def _load_set(subjects, n_readers=tf.data.experimental.AUTOTUNE,
              n_parallel_calls=tf.data.experimental.AUTOTUNE,
              batch_size=100, expand_dim=False):
    path_files = [os.path.join(TFRECORD_FILES_DIR, subject, file_name)
                  for subject in subjects
                  for file_name in sorted(os.listdir(os.path.join(TFRECORD_FILES_DIR, subject)))]
    dataset = tf.data.Dataset.list_files(path_files)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TFRecordDataset(filepath, compression_type="GZIP"),
        cycle_length=n_readers, num_parallel_calls=n_parallel_calls)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda r: _preprocess(r, expand_dim=expand_dim),
                          num_parallel_calls=n_parallel_calls)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


@tf.function
def _preprocess(serialized_eeg_records, expand_dim=False):
    n_channels = 64
    feature_description = {
        "X": tf.io.FixedLenFeature([n_channels], tf.float32),
        "y": tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_eeg_records = tf.io.parse_example(serialized_eeg_records, feature_description)
    X = parsed_eeg_records["X"]
    y = parsed_eeg_records["y"]
    if expand_dim:
        X = X[..., np.newaxis]
    return X, y
