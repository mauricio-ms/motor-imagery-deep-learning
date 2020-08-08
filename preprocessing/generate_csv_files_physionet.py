import os
import shutil
import numpy as np
import pandas as pd
import pyedflib
import re
import functools
import operator
from main import ROOT_DIR


def get_event(event, run_execution):
    if event == "T0":
        return 0

    movement_single_member_runs = [3, 4, 7, 8, 11, 12]
    movement_both_members_runs = [5, 6, 9, 10, 13, 14]

    if event == "T1":
        if run_execution in movement_single_member_runs:
            return 1
        elif run_execution in movement_both_members_runs:
            return 3

    if event == "T2":
        if run_execution in movement_single_member_runs:
            return 2
        elif run_execution in movement_both_members_runs:
            return 4


FREQUENCY = 160
PHYSIONET_DIR = os.path.join(ROOT_DIR, "data/physionet")
RAW_CSV_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-csv-files")

if os.path.exists(RAW_CSV_FILES_DIR):
    shutil.rmtree(RAW_CSV_FILES_DIR)
os.makedirs(RAW_CSV_FILES_DIR)

subjects = sorted(os.listdir(os.path.join(PHYSIONET_DIR, "edf-files")))
#subjects = ["S088"]
for subject in filter(lambda f: re.match("S(\\d+)", f), subjects):
    csv_subject_directory = os.path.join(RAW_CSV_FILES_DIR, subject)
    if not os.path.exists(csv_subject_directory):
        os.makedirs(csv_subject_directory)

    edf_subject_directory = os.path.join(PHYSIONET_DIR, "edf-files", subject)
    edf_files_names = sorted(os.listdir(edf_subject_directory))
    #edf_files_names = ["S088R03.edf"]
    for edf_file_name in filter(lambda f: f.endswith(".edf"), edf_files_names):
        print(f"Converting EDF file {edf_file_name} to CSV ...", end="\r")
        groups_edf_file = re.match("S(\\d+)R(\\d+).edf", edf_file_name).groups()
        run_execution = groups_edf_file[1]

        edf_file = pyedflib.EdfReader(os.path.join(edf_subject_directory, edf_file_name))

        annotations = edf_file.readAnnotations()
        onset_events = annotations[0]
        duration_events = annotations[1]
        events = annotations[2]

        frequency = edf_file.getSampleFrequencies()[0]
        n_samples = int(np.round(np.sum(duration_events), decimals=2) * frequency)
        n_channels = edf_file.signals_in_file
        data = np.zeros((n_samples, n_channels+1))
        # Set invalid label to verify skipped samples
        data[:, -1] = -1
        end_index = None
        for index_event in range(len(onset_events)):
            onset_event = onset_events[index_event]
            duration_event = duration_events[index_event]
            event = events[index_event]

            onset_index = int(onset_event * frequency)
            if end_index is not None and onset_index != end_index:
                onset_index = end_index
            end_index = np.minimum(int(np.round(onset_event + duration_event, decimals=2) * frequency), n_samples)
            event_samples = end_index-onset_index

            for ch in np.arange(n_channels):
                data[onset_index:end_index, ch] = edf_file.readSignal(ch, onset_event, event_samples)
            data[onset_index:end_index, -1] = np.repeat(get_event(event, int(run_execution)), event_samples)

        channels_labels = edf_file.getSignalLabels()
        header = functools.reduce(operator.iconcat, [channels_labels, ["LABEL"]], [])
        csv_path_file = os.path.join(csv_subject_directory, f"{subject}R{run_execution}.csv")
        pd.DataFrame(data)\
            .to_csv(csv_path_file, header=header, index=False)
