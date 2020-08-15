import os
from helpers.file_system_helper import recreate_dir
import numpy as np
import pandas as pd
import mne
import re
import functools
import operator


mne.set_log_level("WARNING")

EVENT_REJECTED_TRIAL = "1023"

EVENTS = {
    "768": "NEW_TRIAL",
    "769": "LEFT_HAND",
    "770": "RIGHT_HAND",
    "771": "BOTH_FEET",
    "772": "TONGUE"
}

LABELS = {
    "LEFT_HAND": 1,
    "RIGHT_HAND": 2,
    "BOTH_FEET": 3,
    "TONGUE": 4
}

# TODO - !!!!!!!!!!!! DO NOT USE THIS, INCOMPLETE IMPLEMENTATION

PROJECT_NAME = "motor-imagery-convolutional-recurrent-neural-network"
ROOT_DIR = os.getcwd()[:os.getcwd().index(PROJECT_NAME) + len(PROJECT_NAME)]
BCI_IV_IIA_DIR = os.path.join(ROOT_DIR, "data/bci-iv-iia")
CSV_FILES_DIR = os.path.join(BCI_IV_IIA_DIR, "csv-files")

recreate_dir(CSV_FILES_DIR)

gdf_file_name = "A02T.gdf"
gdf_file = mne.io.read_raw_gdf(os.path.join(BCI_IV_IIA_DIR, "gdf-files", gdf_file_name), preload=True)
groups_gdf_file = re.match("A(\\d+)([ET]).gdf", gdf_file_name).groups()
subject = groups_gdf_file[0]
session_type = groups_gdf_file[1]

annotations = gdf_file.annotations
start_trials_indexes = [event_index for event_index in range(len(annotations.description))
                        if EVENTS.get(annotations.description[event_index]) == "NEW_TRIAL" and
                        EVENTS.get(annotations.description[event_index + 1])]
valid_trials = len(start_trials_indexes)
rejected_trials = np.sum([1 for event_index in range(len(annotations.description))
                          if annotations.description[event_index] == EVENT_REJECTED_TRIAL])

print("Valid Trials: " + str(valid_trials))
print("Rejected Trials: " + str(rejected_trials))
print("Total Trials: " + str(valid_trials + rejected_trials))

indexes_channels_eeg = [index for index, _ in enumerate(filter(lambda ch: "EEG" in ch, gdf_file.ch_names))]
n_channels = len(indexes_channels_eeg)
frequency = gdf_file.info["sfreq"]
n_samples = int(np.sum(annotations.duration[start_trials_indexes]) * frequency)
data = np.zeros((n_samples, n_channels+1))
# Set invalid label to verify skipped samples
data[:, -1] = -1

# TODO - Verify if we will consider the trial as the onset of the start trial to the next start trial
# because the times seems to be incorrect
start = 0
for event_start_trial_index in start_trials_indexes:
    onset_event = annotations.onset[event_start_trial_index]
    duration_event = annotations.duration[event_start_trial_index]
    onset_index = int(onset_event * frequency)
    end_index = int((onset_event + duration_event) * frequency)
    event_samples = end_index-onset_index

    # The event correspondent to the trial is the following the start trial event
    event_index = event_start_trial_index + 1
    event = EVENTS[annotations.description[event_index]]

    # The index 0 returns the data array of the gdf_file
    # The index 1 returns the times array of the gdf_file
    data[start:start+event_samples, indexes_channels_eeg] = gdf_file[indexes_channels_eeg, onset_index:end_index][0].T
    data[start:start+event_samples, -1] = np.repeat(LABELS[event], event_samples)
    start = start + event_samples

channels_labels = np.array(gdf_file.ch_names)[indexes_channels_eeg]
header = functools.reduce(operator.iconcat, [channels_labels, ["LABEL"]], [])
csv_path_file = os.path.join(CSV_FILES_DIR, f"A{subject}{session_type}.csv")
pd.DataFrame(data)\
    .to_csv(csv_path_file, header=header, index=False)
