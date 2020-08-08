import os
import pandas as pd
from main import ROOT_DIR

PHYSIONET_DIR = os.path.join(ROOT_DIR, "data/physionet")
RAW_CSV_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-csv-files")

subjects = sorted(os.listdir(RAW_CSV_FILES_DIR))
for subject in subjects:
    print("Subject: " + subject)
    subject_directory = os.path.join(RAW_CSV_FILES_DIR, subject)
    files_names = sorted(os.listdir(subject_directory))
    for file_name in files_names:
        path_file = os.path.join(subject_directory, file_name)
        data = pd.read_csv(path_file, header=1).values
        if -1 in data[:, -1]:
            print(f"File {file_name} with skipped samples!")
        elif None in data.flatten():
            print(f"File {file_name} with missing data!")
