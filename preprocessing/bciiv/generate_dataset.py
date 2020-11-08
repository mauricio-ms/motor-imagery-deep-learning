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


def generate(dataset_root_dir, events_enum, labels_enum, labels, classes, window_size):
    dataset_dir = os.path.join(dataset_root_dir, "normalized-by-sample",
                               f"window-{window_size}",
                               "-".join(classes).lower())

    fsh.recreate_dir(dataset_dir)
    info = {
        "n_samples_by_file": 0
    }

    subjects_labels_counts = {}
    gdf_files_dir = os.path.join(dataset_root_dir, "gdf-files")
    gdf_file_names = filter(lambda f: re.match(".*.gdf", f), sorted(os.listdir(gdf_files_dir)))
    for gdf_file_name in gdf_file_names:
        gdf_file = mne.io.read_raw_gdf(os.path.join(gdf_files_dir, gdf_file_name), preload=True)
        groups_gdf_file = re.match("(.)(\\d{2})(\\d{0,2})([ET])\\.gdf", gdf_file_name).groups()
        database_prefix = groups_gdf_file[0]
        subject = groups_gdf_file[1]
        session = groups_gdf_file[2]
        session_type = groups_gdf_file[3]

        labels_filepath = os.path.join(gdf_files_dir, "labels", f"{database_prefix}{subject}{session}{session_type}.mat")
        labels_file = io.loadmat(labels_filepath)["classlabel"]

        annotations = gdf_file.annotations
        start_trials_indexes = [event_index for event_index in range(len(annotations.description))
                                if events_enum.get(annotations.description[event_index]) == "NEW_TRIAL"]

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
            if events_enum[annotations.description[cue_event_index]] == "REJECTED_TRIAL":
                rejected_trials += 1
                continue

            label = labels_enum[labels_file[n_trial][0]]
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

        tfrecord_filepath = os.path.join(dataset_dir, f"{database_prefix}{subject}{session}{session_type}.tfrecord")
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

        labels_counts = np.unique(y, return_counts=True)[1]
        if subject not in subjects_labels_counts:
            subjects_labels_counts[subject] = labels_counts
        else:
            subjects_labels_counts[subject] = subjects_labels_counts[subject] + labels_counts

        print("Info from file " + gdf_file_name)
        print(f"Labels counts: {labels_counts}")
        print("Valid Trials: " + str(valid_trials))
        print("Ignored Trials: " + str(ignored_trials))
        print("Rejected Trials: " + str(rejected_trials))
        print("Total Trials: " + str(valid_trials + ignored_trials + rejected_trials))

    print("Subjects Labels Counts:")
    print(subjects_labels_counts)

    print("Generation dataset ended, saving info data ...")
    print("info[n_samples_by_file]=", info["n_samples_by_file"])
    info_filepath = os.path.join(dataset_dir, "info.pkl")
    with open(info_filepath, "wb") as fp:
        pickle.dump(info, fp, protocol=4)
