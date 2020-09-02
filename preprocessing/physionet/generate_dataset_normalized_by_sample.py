import os
import re
import tensorflow as tf
import numpy as np
import helpers.file_system_helper as fsh
from preprocessing.physionet.config import PHYSIONET_DIR, DAMAGED_SUBJECTS, EXECUTIONS_OF_INTEREST
from preprocessing.physionet.EdfFile import EdfFile
from tensorflow.core.example.feature_pb2 import Features, Feature, FloatList, Int64List
from tensorflow.core.example.example_pb2 import Example

RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")
TFRECORD_FILEPATH = os.path.join(PHYSIONET_DIR, "dataset-normalized-by-sample.tfrecord")
EXECUTIONS_OF_INTEREST_REGEX = f"({'|'.join(EXECUTIONS_OF_INTEREST)})"
CLASSES = {
    "eyes-closed": 0,
    "left-fist": 1,
    "right-fist": 2,
    "both-fists": 3,
    "both-feet": 4
}

fsh.rm_if_exists(TFRECORD_FILEPATH)

print("Gathering samples indexes of interest")
files_samples_indexes = []
subjects = filter(lambda s: s not in DAMAGED_SUBJECTS, sorted(os.listdir(RAW_EDF_FILES_DIR)))
for subject in filter(lambda f: re.match("S(\\d+)", f), subjects):
    edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
    edf_file_names = sorted(os.listdir(edf_subject_path_dir))
    for edf_file_name in filter(lambda f: re.match(f"^{subject}{EXECUTIONS_OF_INTEREST_REGEX}\\.edf$", f),
                                edf_file_names):
        print(f"Gathering samples indexes from the file {edf_file_name} ...", end="\r")
        edf_file = EdfFile(edf_subject_path_dir, edf_file_name)
        for n_sample in range(edf_file.n_samples):
            y = edf_file.labels[n_sample]
            # The data corresponding to the rest time in motor imagery tasks should be ignored
            if y != "rest":
                files_samples_indexes.append((edf_file.data[n_sample], y))
        edf_file.close()

print("Shuffling samples")
shuffled_indexes = np.array(range(len(files_samples_indexes)))
np.random.shuffle(shuffled_indexes)
files_samples_indexes = list(np.array(files_samples_indexes)[shuffled_indexes])
del shuffled_indexes

print("Normalizing and writing TFRecords")
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(TFRECORD_FILEPATH, options) as tfr:
    while len(files_samples_indexes) > 0:
        print(f"{len(files_samples_indexes)} samples remaining ...", end="\r")
        X, y = files_samples_indexes.pop(0)
        X = (X - np.mean(X)) / np.std(X)
        y = CLASSES[y]

        eeg_example = Example(
            features=Features(
                feature={
                    "X": Feature(float_list=FloatList(value=X)),
                    "y": Feature(int64_list=Int64List(value=[y]))
                }
            )
        )
        tfr.write(eeg_example.SerializeToString())
