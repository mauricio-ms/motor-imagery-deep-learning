import os
import re
import tensorflow as tf
import numpy as np
import helpers.file_system_helper as fsh
from preprocessing.physionet.preprocessing_utils import convert_to_2D
from preprocessing.physionet.config import PHYSIONET_DIR, DAMAGED_SUBJECTS, EXECUTIONS_OF_INTEREST
from preprocessing.physionet.EdfFile import EdfFile
from tensorflow.core.example.feature_pb2 import Features, Feature, BytesList, Int64List
from tensorflow.core.example.example_pb2 import Example

RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")
PREPROCESSED_TFRECORD_FILES_DIR = os.path.join(PHYSIONET_DIR, "2D", "tfrecord-files",
                                               "normalized-by-sample", "-".join(EXECUTIONS_OF_INTEREST))
EXECUTIONS_OF_INTEREST_REGEX = f"({'|'.join(EXECUTIONS_OF_INTEREST)})"
CLASSES = {
    "eyes-closed": 0,
    "left-fist": 1,
    "right-fist": 2,
    "both-fists": 3,
    "both-feet": 4
}

fsh.recreate_dir(PREPROCESSED_TFRECORD_FILES_DIR)

subjects = filter(lambda s: s not in DAMAGED_SUBJECTS, sorted(os.listdir(RAW_EDF_FILES_DIR)))
for subject in filter(lambda f: re.match("S(\\d+)", f), subjects):
    print(f"\rGenerating TFRecord file from the subject {subject} ...", end="")

    X_data = np.empty((0, 64))
    y_data = np.empty(0)

    edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
    edf_file_names = sorted(os.listdir(edf_subject_path_dir))
    for edf_file_name in filter(lambda f: re.match(f"^{subject}{EXECUTIONS_OF_INTEREST_REGEX}\\.edf$", f),
                                edf_file_names):
        edf_file = EdfFile(edf_subject_path_dir, edf_file_name)
        samples_selection = edf_file.labels != "rest"
        X_data = np.vstack((X_data, edf_file.data[samples_selection]))
        y_data = np.append(y_data, edf_file.labels[samples_selection])
        edf_file.close()

    tfrecord_subject_filepath = os.path.join(PREPROCESSED_TFRECORD_FILES_DIR, f"{subject}.tfrecord")
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(tfrecord_subject_filepath, options) as writer:
        for n_sample in range(len(y_data)):
            X = X_data[n_sample]
            X = (X - np.mean(X)) / np.std(X)
            X = convert_to_2D(X)
            y = CLASSES[y_data[n_sample]]

            eeg_example = Example(
                features=Features(
                    feature={
                        "X": Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(X).numpy()])),
                        "y": Feature(int64_list=Int64List(value=[y]))
                    }
                )
            )
            writer.write(eeg_example.SerializeToString())
