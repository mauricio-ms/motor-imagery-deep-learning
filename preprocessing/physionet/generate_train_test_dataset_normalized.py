import os
import re
import pickle
import numpy as np
import helpers.file_system_helper as fsh
from preprocessing.physionet.config import PHYSIONET_DIR, EXECUTIONS_OF_INTEREST
from preprocessing.physionet.EdfFile import EdfFile

RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")
PKL_STATISTICS_FILEPATH = os.path.join(PHYSIONET_DIR, "dataset-statistics.pkl")
PKL_TRAIN_DATASET_FILEPATH = os.path.join(PHYSIONET_DIR, "normalized-train-dataset.pkl")
PKL_TEST_DATASET_FILEPATH = os.path.join(PHYSIONET_DIR, "normalized-test-dataset.pkl")
EXECUTIONS_OF_INTEREST_REGEX = f"({'|'.join(EXECUTIONS_OF_INTEREST)})"
CLASSES = {
    "eyes-closed": 0,
    "left-fist": 1,
    "right-fist": 2,
    "both-fists": 3,
    "both-feet": 4
}

fsh.rm_if_exists(PKL_TRAIN_DATASET_FILEPATH)
fsh.rm_if_exists(PKL_TEST_DATASET_FILEPATH)

with open(PKL_STATISTICS_FILEPATH, "rb") as fp:
    statistics = pickle.load(fp)

print("Collecting training data")
X_train = []
y_train = []
for subject in statistics["train_subjects"]:
    edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
    edf_file_names = sorted(os.listdir(edf_subject_path_dir))
    for edf_file_name in filter(lambda f: re.match(f"^{subject}{EXECUTIONS_OF_INTEREST_REGEX}\\.edf$", f),
                                edf_file_names):
        print(f"Gathering data from the file {edf_file_name} ...", end="\r")
        edf_file = EdfFile(edf_subject_path_dir, edf_file_name)
        for n_sample in range(edf_file.n_samples):
            y = edf_file.labels[n_sample]
            # The data corresponding to the rest time in motor imagery tasks should be ignored
            if y != "rest":
                X = edf_file.data[n_sample]
                X = (X - statistics["means"]) / statistics["stds"]
                X_train.append(X)
                y_train.append(CLASSES[y])
        edf_file.close()

shuffled_indexes = np.arange(len(y_train))
np.random.shuffle(shuffled_indexes)
X_train = np.array(X_train)[shuffled_indexes]
y_train = np.array(y_train)[shuffled_indexes]

train_dataset = {
    "X": X_train,
    "y": y_train
}
with open(PKL_TRAIN_DATASET_FILEPATH, "wb") as fp:
    pickle.dump(train_dataset, fp, protocol=4)
del X_train, y_train, train_dataset, shuffled_indexes

print("Collecting test data")
X_test = []
y_test = []
for subject in statistics["test_subjects"]:
    edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
    edf_file_names = sorted(os.listdir(edf_subject_path_dir))
    for edf_file_name in filter(lambda f: re.match(f"^{subject}{EXECUTIONS_OF_INTEREST_REGEX}\\.edf$", f),
                                edf_file_names):
        print(f"Gathering data from the file {edf_file_name} ...", end="\r")
        edf_file = EdfFile(edf_subject_path_dir, edf_file_name)
        for n_sample in range(edf_file.n_samples):
            y = edf_file.labels[n_sample]
            # The data corresponding to the rest time in motor imagery tasks should be ignored
            if y != "rest":
                X = edf_file.data[n_sample]
                X = (X - statistics["means"]) / statistics["stds"]
                X_test.append(X)
                y_test.append(CLASSES[y])
        edf_file.close()

shuffled_indexes = np.arange(len(y_test))
np.random.shuffle(shuffled_indexes)
X_test = np.array(X_test)[shuffled_indexes]
y_test = np.array(y_test)[shuffled_indexes]

test_dataset = {
    "X": X_test,
    "y": y_test
}
with open(PKL_TEST_DATASET_FILEPATH, "wb") as fp:
    pickle.dump(test_dataset, fp, protocol=4)
