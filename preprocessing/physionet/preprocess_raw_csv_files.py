import os
from main import ROOT_DIR
import helpers.file_system_helper as fsh
import pandas as pd
import numpy as np

PHYSIONET_DIR = os.path.join(ROOT_DIR, "data/physionet")
RAW_CSV_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-csv-files")
PREPROCESSED_CSV_FILES_DIR = os.path.join(PHYSIONET_DIR, "preprocessed-csv-files")

fsh.recreate_dir(PREPROCESSED_CSV_FILES_DIR)

subjects = sorted(os.listdir(RAW_CSV_FILES_DIR))
for subject in subjects:
    preprocessed_csv_files_subject_path_dir = os.path.join(PREPROCESSED_CSV_FILES_DIR, subject)
    fsh.mkdir_if_not_exists(preprocessed_csv_files_subject_path_dir)

    raw_csv_subject_path_dir = os.path.join(RAW_CSV_FILES_DIR, subject)
    csv_files_names = sorted(os.listdir(raw_csv_subject_path_dir))
    for csv_file_name in csv_files_names:
        print(f"Pre-processing raw CSV file {csv_file_name} ...", end="\r")
        raw_csv_path_file = os.path.join(raw_csv_subject_path_dir, csv_file_name)
        df = pd.read_csv(raw_csv_path_file)
        data = df.values
        header = df.columns.values
        eeg_data = data[:, :-1]
        eeg_data_normalized = (eeg_data - np.mean(eeg_data, axis=0))/np.std(eeg_data, axis=0)
        labels = data[:, -1]
        data_normalized = np.column_stack([eeg_data_normalized, labels])

        preprocessed_csv_path_file = os.path.join(preprocessed_csv_files_subject_path_dir, csv_file_name)
        pd.DataFrame(data_normalized) \
            .to_csv(preprocessed_csv_path_file, header=header, index=False)
