import os
import re
import functools
import operator
import numpy as np
import pandas as pd
import helpers.file_system_helper as fsh
from preprocessing.physionet.config import PHYSIONET_DIR, DAMAGED_SUBJECTS, EXECUTIONS_OF_INTEREST
from preprocessing.physionet.EdfFile import EdfFile

RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")
RAW_CSV_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-csv-files")
EXECUTIONS_OF_INTEREST_REGEX = f"({'|'.join(EXECUTIONS_OF_INTEREST)})"

fsh.recreate_dir(RAW_CSV_FILES_DIR)

subjects = filter(lambda s: s not in DAMAGED_SUBJECTS, sorted(os.listdir(RAW_EDF_FILES_DIR)))
for subject in filter(lambda f: re.match("S(\\d+)", f), subjects):
    csv_subject_path_dir = os.path.join(RAW_CSV_FILES_DIR, subject)
    fsh.mkdir_if_not_exists(csv_subject_path_dir)

    edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
    edf_files_names = sorted(os.listdir(edf_subject_path_dir))
    for edf_file_name in filter(lambda f: re.match(f"^{subject}{EXECUTIONS_OF_INTEREST_REGEX}\\.edf$", f),
                                edf_files_names):
        print(f"Generating CSV file from the file {edf_file_name} ...", end="\r")
        edf_file = EdfFile(edf_subject_path_dir, edf_file_name)
        data = np.column_stack((edf_file.data, edf_file.labels))

        header = functools.reduce(operator.iconcat, [edf_file.channels_labels, ["LABEL"]], [])
        csv_path_file = edf_file.get_path_file(csv_subject_path_dir, "csv")
        pd.DataFrame(data)\
            .to_csv(csv_path_file, header=header, index=False)

        edf_file.close()
