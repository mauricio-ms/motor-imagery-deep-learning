import os
from main import ROOT_DIR
from logger.log_factory import get_logger
import tensorflow as tf
import numpy as np

LOGGER = get_logger("physionet.py")
N_CHANNELS = 64
PHYSIONET_DIR = os.path.join(ROOT_DIR, "data/physionet")
CSV_FILES_DIR = os.path.join(PHYSIONET_DIR, "preprocessed-csv-files")


def load_data(train_size=0.75, validation_size=None, n_subjects=None, **kwargs):
    LOGGER.info("Loading Physionet dataset ...")
    subjects = np.array(sorted(os.listdir(CSV_FILES_DIR)))
    if n_subjects is not None:
        subjects = subjects[:n_subjects]
    train_subjects, test_subjects = _train_test_split_subjects(subjects, train_size)
    if validation_size is not None:
        train_subjects, validation_subjects = _train_test_split_subjects(train_subjects, 1-validation_size)
        LOGGER.info(f"(Train, Validation, Test) Subjects = "
                    f"({len(train_subjects)}, {len(validation_subjects)}, {len(test_subjects)})")
        return _load_set(train_subjects, **kwargs), \
            _load_set(validation_subjects, **kwargs), \
            _load_set(test_subjects, **kwargs)

    LOGGER.info(f"(Train, Validation) Subjects = ({len(train_subjects)}, {len(test_subjects)})")
    return _load_set(train_subjects, **kwargs), _load_set(test_subjects, **kwargs)


def _train_test_split_subjects(subjects, train_size):
    train_subjects_mask = np.random.rand(len(subjects)) < train_size
    return subjects[train_subjects_mask], subjects[~train_subjects_mask]


def _load_set(subjects, n_readers=5, n_parse_threads=5, batch_size=100,
              convert_to_2d=False, expand_dim=False):
    path_files = [os.path.join(CSV_FILES_DIR, subject, file_name)
                  for subject in subjects
                  for file_name in sorted(os.listdir(os.path.join(CSV_FILES_DIR, subject)))]
    dataset = tf.data.Dataset.list_files(path_files)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers)
    dataset = dataset.map(lambda r: _preprocess(r, convert_to_2d=convert_to_2d,
                                                expand_dim=expand_dim),
                          num_parallel_calls=n_parse_threads)
    return dataset.batch(batch_size).prefetch(1)


def _preprocess(eeg_record, convert_to_2d=False, expand_dim=False):
    # Create the definitions for the columns (channels + label)
    # The empty array tells TensorFlow to raise exception to missing values
    defs = [tf.constant([], dtype=tf.float32)] * (N_CHANNELS + 1)
    fields = tf.io.decode_csv(eeg_record, record_defaults=defs)
    x = _get_features(fields, convert_to_2d=convert_to_2d)
    if expand_dim:
        x = x[..., np.newaxis]
    y = tf.cast(tf.stack(fields[-1:]), tf.int32)
    return x, y


def _get_features(fields, convert_to_2d=False):
    if convert_to_2d:
        x = fields[:-1]
        # TODO Test Variable e tf.map_fn, tf.SparseTensor, tf.TensorArray, page 383 hands-on
        # x = np.zeros((64))
        # return tf.constant([
        #     [0, 0, 0, 0, x[21], x[22], x[23], 0, 0, 0, 0],
        #     [0, 0, 0, x[24], x[25], x[26], x[27], x[28], 0, 0, 0],
        #     [0, x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37], 0],
        #     [0, x[38], x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[39], 0],
        #     [x[42], x[40], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[41], x[43]],
        #     [0, x[44], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[45], 0],
        #     [0, x[46], x[47], x[48], x[49], x[50], x[51], x[52], x[53], x[54], 0],
        #     [0, 0, 0, x[55], x[56], x[57], x[58], x[59], 0, 0, 0],
        #     [0, 0, 0, 0, x[60], x[61], x[62], 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, x[63], 0, 0, 0, 0, 0]
        # ])
        # x_2d = np.zeros((10, 11))
        # x_2d[0, :] = (0, 0, 0, 0, x[21], x[22], x[23], 0, 0, 0, 0)
        # x_2d[1, :] = (0, 0, 0, x[24], x[25], x[26], x[27], x[28], 0, 0, 0)
        # x_2d[2, :] = (0, x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37], 0)
        # x_2d[3, :] = (0, x[38], x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[39], 0)
        # x_2d[4, :] = (x[42], x[40], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[41], x[43])
        # x_2d[5, :] = (0, x[44], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[45], 0)
        # x_2d[6, :] = (0, x[46], x[47], x[48], x[49], x[50], x[51], x[52], x[53], x[54], 0)
        # x_2d[7, :] = (0, 0, 0, x[55], x[56], x[57], x[58], x[59], 0, 0, 0)
        # x_2d[8, :] = (0, 0, 0, 0, x[60], x[61], x[62], 0, 0, 0, 0)
        # x_2d[9, :] = (0, 0, 0, 0, 0, x[63], 0, 0, 0, 0, 0)
        # return tf.stack(x_2d)
    return tf.stack(fields[:-1])
