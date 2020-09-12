import os
import re
import tensorflow as tf
import numpy as np
from operator import itemgetter
from itertools import groupby
import helpers.file_system_helper as fsh
from preprocessing.physionet.preprocessing_utils import convert_to_2D
from preprocessing.physionet.config import PHYSIONET_DIR
from preprocessing.physionet.EdfFile import EdfFile
from tensorflow.core.example.feature_pb2 import Features, Feature, BytesList, Int64List
from tensorflow.core.example.example_pb2 import Example


def windows(data, size):
    start = 0
    while (start+size) < len(data):
        yield int(start), int(start + size)
        start += (size / 2)


RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")
DATASET_DIR = os.path.join(PHYSIONET_DIR, "3D", "tfrecord-files", "left-right")
DAMAGED_SUBJECTS = ["S088", "S089", "S092", "S100"]
EXECUTIONS_OF_INTEREST = ["R04", "R08", "R12"]
EXECUTIONS_OF_INTEREST_REGEX = f"({'|'.join(EXECUTIONS_OF_INTEREST)})"
CLASSES = {
    "eyes-closed": 0,
    "left-fist": 1,
    "right-fist": 2,
    "both-fists": 3,
    "both-feet": 4
}

fsh.recreate_dir(DATASET_DIR)

subjects = filter(lambda s: s not in DAMAGED_SUBJECTS, sorted(os.listdir(RAW_EDF_FILES_DIR)))
for subject in filter(lambda f: re.match("S(\\d+)", f), subjects):
    print(f"\rGenerating TFRecord file from the subject {subject} ...", end="")

    X_segments = np.empty((0, 64))
    y = np.empty(0, dtype=np.int64)

    edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
    edf_file_names = sorted(os.listdir(edf_subject_path_dir))
    for edf_file_name in filter(lambda f: re.match(f"^{subject}{EXECUTIONS_OF_INTEREST_REGEX}\\.edf$", f),
                                edf_file_names):
        edf_file = EdfFile(edf_subject_path_dir, edf_file_name)
        events_windows = [next(group) for key, group in groupby(enumerate(edf_file.labels), key=itemgetter(1))]
        n_events = len(events_windows)
        for index, (event_start_index, event) in enumerate(events_windows):
            if event == "rest":
                continue

            X = edf_file.data[event_start_index:] if index+1 == n_events \
                else edf_file.data[event_start_index:events_windows[index+1][0]]
            for (start_segment, end_segment) in windows(X, 10):
                X_segments = np.vstack((X_segments, X[start_segment:end_segment]))
                y = np.append(y, CLASSES[event])

        edf_file.close()

    X_segments = X_segments.reshape((-1, 10, 64))
    tfrecord_subject_filepath = os.path.join(DATASET_DIR, f"{subject}.tfrecord")
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(tfrecord_subject_filepath, options) as writer:
        for n_segment in range(len(y)):
            X_segment = np.array(list(map(lambda x: convert_to_2D((x - np.mean(x)) / np.std(x)),
                                          X_segments[n_segment])))

            eeg_example = Example(
                features=Features(
                    feature={
                        "X": Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(X_segment).numpy()])),
                        "y": Feature(int64_list=Int64List(value=[y[n_segment]]))
                    }
                )
            )
            writer.write(eeg_example.SerializeToString())
