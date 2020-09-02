import os
import re
import pickle
import helpers.file_system_helper as fsh
from preprocessing.physionet.config import PHYSIONET_DIR, DAMAGED_SUBJECTS, EXECUTIONS_OF_INTEREST
from preprocessing.physionet.EdfFile import EdfFile

RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")
PKL_DATA_FILEPATH = os.path.join(PHYSIONET_DIR, "dataset.pkl")
PKL_LABELS_FILEPATH = os.path.join(PHYSIONET_DIR, "dataset-labels.pkl")
EXECUTIONS_OF_INTEREST_REGEX = f"({'|'.join(EXECUTIONS_OF_INTEREST)})"
CLASSES = {
    "eyes-closed": 0,
    "left-fist": 1,
    "right-fist": 2,
    "both-fists": 3,
    "both-feet": 4
}

fsh.rm_if_exists(PKL_DATA_FILEPATH)
fsh.rm_if_exists(PKL_LABELS_FILEPATH)

eeg_data = []
labels = []
subjects = filter(lambda s: s not in DAMAGED_SUBJECTS, sorted(os.listdir(RAW_EDF_FILES_DIR)))
for subject in filter(lambda f: re.match("S(\\d+)", f), subjects):
    edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
    edf_file_names = sorted(os.listdir(edf_subject_path_dir))
    for edf_file_name in filter(lambda f: re.match(f"^{subject}{EXECUTIONS_OF_INTEREST_REGEX}\\.edf$", f),
                                edf_file_names):
        print(f"Gathering data from the file {edf_file_name} ...", end="\r")
        edf_file = EdfFile(edf_subject_path_dir, edf_file_name)
        for n_sample in range(edf_file.n_samples):
            y = edf_file.labels[n_sample]
            # The data corresponding to the rest time in motor imagery tasks should be ignored
            if y != -1:
                eeg_data.append(edf_file.data[n_sample])
                labels.append(CLASSES[y])
        edf_file.close()

with open(PKL_DATA_FILEPATH, "wb") as fp:
    pickle.dump(eeg_data, fp, protocol=4)

with open(PKL_LABELS_FILEPATH, "wb") as fp:
    pickle.dump(labels, fp, protocol=4)
