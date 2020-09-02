import os
import re
import numpy as np
import pickle
import helpers.file_system_helper as fsh
from preprocessing.physionet.config import PHYSIONET_DIR, DAMAGED_SUBJECTS, EXECUTIONS_OF_INTEREST
from preprocessing.physionet.EdfFile import EdfFile

PKL_STATISTICS_FILEPATH = os.path.join(PHYSIONET_DIR, "dataset-statistics.pkl")
RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")
EXECUTIONS_OF_INTEREST_REGEX = f"({'|'.join(EXECUTIONS_OF_INTEREST)})"
TRAIN_SIZE = 0.75
N_CHANNELS = 64

fsh.rm_if_exists(PKL_STATISTICS_FILEPATH)

subjects = filter(lambda s: s not in DAMAGED_SUBJECTS and re.match("S(\\d+)", s),
                  sorted(os.listdir(RAW_EDF_FILES_DIR)))
subjects = np.array(list(subjects))
train_subjects_mask = np.random.rand(len(subjects)) < TRAIN_SIZE
train_subjects = subjects[train_subjects_mask]
test_subjects = subjects[~train_subjects_mask]

means = np.empty([N_CHANNELS])
stds = np.empty([N_CHANNELS])

for ch in range(N_CHANNELS):
    eeg_data = np.empty([0, 1])
    for subject in subjects:
        print(f"Collecting data from the channel {ch} for the subject {subject} ...", end="\r")
        edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
        edf_file_names = sorted(os.listdir(edf_subject_path_dir))
        for edf_file_name in filter(lambda f: re.match(f"^{subject}{EXECUTIONS_OF_INTEREST_REGEX}\\.edf$", f),
                                    edf_file_names):
            edf_file = EdfFile(edf_subject_path_dir, edf_file_name, channels=[ch])
            samples_selection = edf_file.labels != "rest"
            eeg_data = np.vstack([eeg_data, edf_file.data[samples_selection]])
            edf_file.close()

    means[ch] = np.mean(eeg_data)
    stds[ch] = np.std(eeg_data)

statistics = {
    "train_size": TRAIN_SIZE,
    "train_subjects": train_subjects,
    "test_subjects": test_subjects,
    "means": means,
    "stds": stds
}

with open(PKL_STATISTICS_FILEPATH, "wb") as fp:
    pickle.dump(statistics, fp, protocol=4)
