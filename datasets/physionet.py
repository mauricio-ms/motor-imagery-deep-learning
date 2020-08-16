import os
from main import ROOT_DIR
from logger.log_factory import get_logger
import tensorflow as tf
import numpy as np

LOGGER = get_logger("physionet.py")
N_CHANNELS = 64
FEATURE_DESCRIPTION = {
    "X": tf.io.FixedLenFeature([N_CHANNELS], tf.float32),
    "y": tf.io.FixedLenFeature([], tf.int64)
}
PHYSIONET_DIR = os.path.join(ROOT_DIR, "data/physionet")
TFRECORD_FILES_DIR = os.path.join(PHYSIONET_DIR, "preprocessed-tfrecord-files")


def load_data(train_size=0.75, validation_size=None, n_subjects=None, **kwargs):
    LOGGER.info("Loading Physionet dataset ...")
    subjects = np.array(sorted(os.listdir(TFRECORD_FILES_DIR)))
    if n_subjects is not None:
        np.random.shuffle(subjects)
        subjects = subjects[:n_subjects]
    train_subjects, test_subjects = _train_test_split_subjects(subjects, train_size)
    if validation_size is not None:
        train_subjects, validation_subjects = _train_test_split_subjects(train_subjects, 1-validation_size)
        LOGGER.info(f"(Train, Validation, Test) Subjects = "
                    f"({len(train_subjects)}, {len(validation_subjects)}, {len(test_subjects)})")
        return _load_set(train_subjects, **kwargs), \
            _load_set(validation_subjects, **kwargs), \
            _load_set(test_subjects, **kwargs)

    LOGGER.info(f"(Train, Test) Subjects = ({len(train_subjects)}, {len(test_subjects)})")
    return _load_set(train_subjects, **kwargs), _load_set(test_subjects, **kwargs)


def _train_test_split_subjects(subjects, train_size):
    train_subjects_mask = np.random.rand(len(subjects)) < train_size
    return subjects[train_subjects_mask], subjects[~train_subjects_mask]


# TODO - convert_to_2d should define the data directory
def _load_set(subjects, n_readers=5, n_parse_threads=5, batch_size=100,
              convert_to_2d=False, expand_dim=False):
    path_files = [os.path.join(TFRECORD_FILES_DIR, subject, file_name)
                  for subject in subjects
                  for file_name in sorted(os.listdir(os.path.join(TFRECORD_FILES_DIR, subject)))]
    dataset = tf.data.Dataset.list_files(path_files)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TFRecordDataset(path_files, compression_type="GZIP"),
        cycle_length=n_readers)
    dataset = dataset.map(lambda r: _preprocess(r, expand_dim=expand_dim),
                          num_parallel_calls=n_parse_threads)
    return dataset.batch(batch_size).prefetch(1)


def _preprocess(serialized_eeg_record, expand_dim=False):
    parsed_eeg_record = tf.io.parse_single_example(serialized_eeg_record, FEATURE_DESCRIPTION)
    X = parsed_eeg_record["X"]
    y = parsed_eeg_record["y"]
    if expand_dim:
        X = X[..., np.newaxis]
    return X, y
