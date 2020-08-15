import os
import re
import pandas as pd
import functools
import operator
import helpers.file_system_helper as fsh
from preprocessing.physionet.config import PHYSIONET_DIR, DAMAGED_SUBJECTS
from preprocessing.physionet.EdfFile import EdfFile

RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")
RAW_CSV_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-csv-files")

fsh.recreate_dir(RAW_CSV_FILES_DIR)

# Ignore subjects with damaged data
subjects = filter(lambda s: s not in DAMAGED_SUBJECTS, sorted(os.listdir(RAW_EDF_FILES_DIR)))
for subject in filter(lambda f: re.match("S(\\d+)", f), subjects):
    csv_subject_path_dir = os.path.join(RAW_CSV_FILES_DIR, subject)
    fsh.mkdir_if_not_exists(csv_subject_path_dir)

    edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
    edf_files_names = sorted(os.listdir(edf_subject_path_dir))
    for edf_file_name in filter(lambda f: f.endswith(".edf"), edf_files_names):
        print(f"Generating CSV file from the file {edf_file_name} ...", end="\r")
        edf_file = EdfFile(edf_subject_path_dir, edf_file_name)

        header = functools.reduce(operator.iconcat, [edf_file.channels_labels, ["LABEL"]], [])
        csv_path_file = edf_file.get_path_file(csv_subject_path_dir, "csv")
        pd.DataFrame(edf_file.data)\
            .to_csv(csv_path_file, header=header, index=False)
