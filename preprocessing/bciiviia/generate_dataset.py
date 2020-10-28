import os
import pickle
import re

import mne
import numpy as np
import tensorflow as tf
from scipy import io
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, Feature, BytesList, Int64List

import helpers.file_system_helper as fsh
from main import ROOT_DIR

mne.set_log_level("WARNING")

EVENTS_ENUM = {
    "768": "NEW_TRIAL",
    "769": "LEFT_HAND",
    "770": "RIGHT_HAND",
    "771": "BOTH_FEET",
    "772": "TONGUE",
    "783": "CUE_UNKNOWN",
    "1023": "REJECTED_TRIAL"
}

LABELS_ENUM = {
    1: "LEFT_HAND",
    2: "RIGHT_HAND",
    3: "BOTH_FEET",
    4: "TONGUE"
}

BCI_IV_IIA_DIR = os.path.join(ROOT_DIR, "data/bci-iv-iia")
GDF_FILES_DIR = os.path.join(BCI_IV_IIA_DIR, "gdf-files")


def generate(window_size=1000, classes=None):
    if classes is None:
        classes = ["LEFT_HAND", "RIGHT_HAND"]

    dataset_dir = os.path.join(BCI_IV_IIA_DIR, "normalized-by-sample",
                               f"window-{window_size}",
                               "-".join(classes).lower())

    label_value = 0
    labels = {}
    if "LEFT_HAND" in classes:
        labels["LEFT_HAND"] = label_value
        label_value += 1
    if "RIGHT_HAND" in classes:
        labels["RIGHT_HAND"] = label_value
        label_value += 1
    if "BOTH_FEET" in classes:
        labels["BOTH_FEET"] = label_value
        label_value += 1
    if "TONGUE" in classes:
        labels["TONGUE"] = label_value
        label_value += 1

    fsh.recreate_dir(dataset_dir)
    info = {
        "n_samples_by_file": 0
    }
    gdf_file_names = filter(lambda f: re.match(".*.gdf", f), sorted(os.listdir(GDF_FILES_DIR)))
    for gdf_file_name in gdf_file_names:
        gdf_file = mne.io.read_raw_gdf(os.path.join(GDF_FILES_DIR, gdf_file_name), preload=True)
        groups_gdf_file = re.match("A(\\d+)([ET]).gdf", gdf_file_name).groups()
        subject = groups_gdf_file[0]
        session_type = groups_gdf_file[1]

        labels_filepath = os.path.join(GDF_FILES_DIR, "labels", f"A{subject}{session_type}.mat")
        labels_file = io.loadmat(labels_filepath)["classlabel"]

        annotations = gdf_file.annotations
        start_trials_indexes = [event_index for event_index in range(len(annotations.description))
                                if EVENTS_ENUM.get(annotations.description[event_index]) == "NEW_TRIAL"]

        indexes_channels_eeg = [index for index, _ in enumerate(filter(lambda ch: "EEG" in ch, gdf_file.ch_names))]
        n_channels = len(indexes_channels_eeg)
        frequency = int(gdf_file.info["sfreq"])
        # should consider the complete motor imagery period (4s),
        # not only the cue exhibition period (1.25s)
        duration_event = window_size // frequency
        n_samples = frequency * duration_event

        rejected_trials = 0
        ignored_trials = 0
        X = np.empty((0, n_channels))
        y = np.empty(0, dtype=np.int64)
        for n_trial, event_start_trial_index in enumerate(start_trials_indexes):
            cue_event_index = event_start_trial_index + 1
            onset_event = annotations.onset[cue_event_index]
            onset_index = int(np.ceil(onset_event * frequency))
            end_index = int(np.ceil((onset_event + duration_event) * frequency))
            event_samples = end_index - onset_index
            if event_samples != n_samples:
                end_index += n_samples - event_samples

            # The event correspondent to the trial is the following the start trial event
            if EVENTS_ENUM[annotations.description[cue_event_index]] == "REJECTED_TRIAL":
                rejected_trials += 1
                continue

            label = LABELS_ENUM[labels_file[n_trial][0]]
            if label not in classes:
                ignored_trials += 1
                continue

            # The index 0 returns the data array of the gdf_file
            # The index 1 returns the times array of the gdf_file
            x = gdf_file[indexes_channels_eeg, onset_index:end_index][0].T
            x = (x - np.mean(x)) / np.std(x)
            X = np.vstack((X, x))

            y = np.append(y, labels[label])

        gdf_file.close()
        X = X.reshape((-1, n_samples, n_channels))

        tfrecord_filepath = os.path.join(dataset_dir, f"A{subject}{session_type}.tfrecord")
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(tfrecord_filepath, options) as writer:
            for n_segment in range(len(y)):
                eeg_example = Example(
                    features=Features(
                        feature={
                            "X": Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(X[n_segment]).numpy()])),
                            "y": Feature(int64_list=Int64List(value=[y[n_segment]]))
                        }
                    )
                )
                writer.write(eeg_example.SerializeToString())

        valid_trials = len(y)
        if valid_trials > info["n_samples_by_file"]:
            info["n_samples_by_file"] = valid_trials

        print("Info from file " + gdf_file_name)
        print("Valid Trials: " + str(valid_trials))
        print("Ignored Trials: " + str(ignored_trials))
        print("Rejected Trials: " + str(rejected_trials))
        print("Total Trials: " + str(valid_trials + ignored_trials + rejected_trials))

    print("Generation dataset ended, saving info data ...")
    print("info[n_samples_by_file]=", info["n_samples_by_file"])
    info_filepath = os.path.join(dataset_dir, "info.pkl")
    with open(info_filepath, "wb") as fp:
        pickle.dump(info, fp, protocol=4)


#generate(classes=["LEFT_HAND", "RIGHT_HAND", "BOTH_FEET", "TONGUE"])
generate(window_size=250)
