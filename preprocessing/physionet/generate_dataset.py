import os
import pickle
import re
from itertools import groupby
from operator import itemgetter

import numpy as np
import tensorflow as tf
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, Feature, BytesList, Int64List

import helpers.file_system_helper as fsh
from preprocessing.physionet.EdfFile import EdfFile
from preprocessing.physionet.config import PHYSIONET_DIR

RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")


def generate(shape, window_size=10, start=0, offset=5, noise_samples=0,
             classes=None, damaged_subjects=None):
    if classes is None:
        classes = ["left-fist", "right-fist"]
    if damaged_subjects is None:
        damaged_subjects = ["S088", "S089", "S092", "S100", "S104"]
    if type(shape) not in (tuple, list):
        shape = [shape]

    dataset_dir = os.path.join(PHYSIONET_DIR, "normalized-by-sample",
                               _get_window_folder_name(window_size, start, offset, noise_samples),
                               f"{window_size}x{'x'.join(str(dim) for dim in shape)}",
                               "-".join(classes))
    executions = []
    if any([c in classes for c in ["eyes-closed"]]):
        executions.append("R02")
    if any([c in classes for c in ["left-fist", "right-fist"]]):
        for execution in ["R04", "R08", "R12"]:
            executions.append(execution)
    if any([c in classes for c in ["both-fists", "both-feet"]]):
        for execution in ["R06", "R10", "R14"]:
            executions.append(execution)

    label_value = 0
    labels = {}
    if "eyes-closed" in classes:
        labels["eyes-closed"] = label_value
        label_value += 1
    if "left-fist" in classes:
        labels["left-fist"] = label_value
        label_value += 1
    if "right-fist" in classes:
        labels["right-fist"] = label_value
        label_value += 1
    if "both-fists" in classes:
        labels["both-fists"] = label_value
        label_value += 1
    if "both-feet" in classes:
        labels["both-feet"] = label_value
        label_value += 1

    fsh.recreate_dir(dataset_dir)

    regex_executions = f"({'|'.join(executions)})"
    info = {
        "n_samples_by_subject": 0
    }
    subjects = filter(lambda s: s not in damaged_subjects, sorted(os.listdir(RAW_EDF_FILES_DIR)))
    for subject in filter(lambda f: re.match("S(\\d+)", f), subjects):
        print(f"Generating TFRecord file from the subject {subject} ...")

        X_segments = np.empty((0, 64))
        y = np.empty(0, dtype=np.int64)

        edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
        edf_file_names = sorted(os.listdir(edf_subject_path_dir))
        for edf_file_name in filter(lambda f: re.match(f"^{subject}{regex_executions}\\.edf$", f),
                                    edf_file_names):
            edf_file = EdfFile(edf_subject_path_dir, edf_file_name)
            events_windows = [next(group) for key, group in groupby(enumerate(edf_file.labels), key=itemgetter(1))]
            n_events = len(events_windows)
            for index, (event_start_index, event) in enumerate(events_windows):
                if event == "rest":
                    continue

                event_start_index += start
                X = edf_file.data[event_start_index:] if index + 1 == n_events \
                    else edf_file.data[event_start_index:events_windows[index + 1][0]]
                n_segments = 0
                for (start_segment, end_segment) in _windows(X, window_size, offset):
                    x_segment = X[start_segment:end_segment]
                    X_segments = np.vstack((X_segments, x_segment))
                    y = np.append(y, labels[event])
                    for _ in range(noise_samples):
                        noise = np.random.normal(0, 1, x_segment.shape)
                        X_segments = np.vstack((X_segments, x_segment + noise))
                        y = np.append(y, labels[event])
                    n_segments += 1
                print(f"X{X.shape} splitted into {n_segments} segments of "
                      f"{window_size} samples with offset of {offset} plus {noise_samples} noise samples")

            edf_file.close()

        if len(y) > info["n_samples_by_subject"]:
            info["n_samples_by_subject"] = len(y)

        print("Labels: ", len(y), len(y[y == 0]), len(y[y == 1]))

        X_segments = X_segments.reshape((-1, window_size, 64))
        print("Has nan: ", np.isnan(X_segments).any())

        tfrecord_subject_filepath = os.path.join(dataset_dir, f"{subject}.tfrecord")
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(tfrecord_subject_filepath, options) as writer:
            for n_segment in range(len(y)):
                X_segment = np.array(list(map(lambda x: _process_record(x, shape), X_segments[n_segment])))

                eeg_example = Example(
                    features=Features(
                        feature={
                            "X": Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(X_segment).numpy()])),
                            "y": Feature(int64_list=Int64List(value=[y[n_segment]]))
                        }
                    )
                )
                writer.write(eeg_example.SerializeToString())

    print("n_samples_by_subject: ", info["n_samples_by_subject"])
    info_filepath = os.path.join(dataset_dir, "info.pkl")
    with open(info_filepath, "wb") as fp:
        pickle.dump(info, fp, protocol=4)


def _get_window_folder_name(window_size, start, offset, noise_samples):
    folder_name = f"window-{window_size}"
    if start > 0:
        folder_name += f"-start-{start}"
    elif start < 0:
        folder_name += f"-start-minus-{np.abs(start)}"
    if offset > 0:
        folder_name += f"-offset-{offset}"
    if noise_samples > 0:
        folder_name += f"-times-{noise_samples}-noise-samples"
    return folder_name


def _windows(data, size, offset):
    start = 0
    while (start + size) <= len(data):
        yield int(start), int(start + size)
        if offset > 0:
            start += offset
        else:
            start += size


def _process_record(x, shape):
    x = (x - np.mean(x)) / np.std(x)
    if shape == (10, 11):
        return _map_to_10x11(x)
    elif shape == (7, 9):
        return _map_to_7x9(x)
    elif shape == [64]:
        return x
    raise AttributeError("Unexpected value for parameter 'shape'.")


def _map_to_10x11(x):
    X = np.zeros([10, 11])

    X[0] = (0, 0, 0, 0, x[21], x[22], x[23], 0, 0, 0, 0)
    X[1] = (0, 0, 0, x[24], x[25], x[26], x[27], x[28], 0, 0, 0)
    X[2] = (0, x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37], 0)
    X[3] = (0, x[38], x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[39], 0)
    X[4] = (x[42], x[40], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[41], x[43])
    X[5] = (0, x[44], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[45], 0)
    X[6] = (0, x[46], x[47], x[48], x[49], x[50], x[51], x[52], x[53], x[54], 0)
    X[7] = (0, 0, 0, x[55], x[56], x[57], x[58], x[59], 0, 0, 0)
    X[8] = (0, 0, 0, 0, x[60], x[61], x[62], 0, 0, 0, 0)
    X[9] = (0, 0, 0, 0, 0, x[63], 0, 0, 0, 0, 0)

    return X


def _map_to_7x9(x):
    X = np.zeros([7, 9])

    X[0] = (x[24], x[21], x[25], x[22], x[26], x[23], x[27], x[28], 0)
    X[1] = (x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37])
    X[2] = (x[38], x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[39])
    X[3] = (x[40], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[41])
    X[4] = (x[44], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[45])
    X[5] = (x[46], x[47], x[48], x[49], x[50], x[51], x[52], x[53], x[54])
    X[6] = (x[55], x[60], x[56], x[63], x[61], x[57], x[62], x[58], x[59])

    return X


generate(shape=64, window_size=480, start=-40, offset=0)
