import os
import re
import numpy as np
import helpers.file_system_helper as fsh
import tensorflow as tf
from tensorflow.core.example.feature_pb2 import Features, Feature, FloatList, Int64List
from tensorflow.core.example.example_pb2 import Example
from preprocessing.physionet.config import PHYSIONET_DIR, DAMAGED_SUBJECTS, EXECUTIONS_OF_INTEREST
from preprocessing.physionet.EdfFile import EdfFile

RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")
RAW_TFRECORD_FILES_DIR = os.path.join(PHYSIONET_DIR, "preprocessed-tfrecord-files-2")
EXECUTIONS_OF_INTEREST_REGEX = f"({'|'.join(EXECUTIONS_OF_INTEREST)})"

fsh.recreate_dir(RAW_TFRECORD_FILES_DIR)

subjects = filter(lambda s: s not in DAMAGED_SUBJECTS, sorted(os.listdir(RAW_EDF_FILES_DIR)))
for subject in filter(lambda f: re.match("S(\\d+)", f), subjects):
    tfrecord_subject_path_dir = os.path.join(RAW_TFRECORD_FILES_DIR, subject)
    fsh.mkdir_if_not_exists(tfrecord_subject_path_dir)

    edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
    edf_file_names = sorted(os.listdir(edf_subject_path_dir))
    for edf_file_name in filter(lambda f: re.match(f"^{subject}{EXECUTIONS_OF_INTEREST_REGEX}\\.edf$", f),
                                edf_file_names):
        print(f"Generating TFRecord file from the file {edf_file_name} ...", end="\r")
        edf_file = EdfFile(edf_subject_path_dir, edf_file_name)

        tfrecord_path_file = edf_file.get_path_file(tfrecord_subject_path_dir, "tfrecord")
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(tfrecord_path_file, options) as tfr:
            for n_sample in range(edf_file.n_samples):
                eeg_record = edf_file.data[n_sample, :-1]
                eeg_record_normalized = (eeg_record - np.mean(eeg_record)) / np.std(eeg_record)
                labels = edf_file.data[n_sample, -1].astype(int)
                eeg_example = Example(
                    features=Features(
                        feature={
                            "X": Feature(float_list=FloatList(value=eeg_record_normalized)),
                            "y": Feature(int64_list=Int64List(value=[labels]))
                        }
                    )
                )
                tfr.write(eeg_example.SerializeToString())
