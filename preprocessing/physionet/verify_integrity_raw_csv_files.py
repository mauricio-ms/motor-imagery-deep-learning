import os
from main import ROOT_DIR
import tensorflow as tf

N_CHANNELS = 64
PHYSIONET_DIR = os.path.join(ROOT_DIR, "data/physionet")
RAW_CSV_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-csv-files")


@tf.function
def _preprocess(eeg_record):
    defs = [tf.constant([], dtype=tf.float32)] * (N_CHANNELS + 1)
    fields = tf.io.decode_csv(eeg_record, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.cast(tf.stack(fields[-1:]), tf.int32)
    return x, y


corrupted_files = []
subjects = sorted(os.listdir(RAW_CSV_FILES_DIR))
for subject in subjects:
    subject_directory = os.path.join(RAW_CSV_FILES_DIR, subject)
    files_names = sorted(os.listdir(subject_directory))
    for file_name in files_names:
        file_path = os.path.join(subject_directory, file_name)
        dataset = tf.data.TextLineDataset(file_path).skip(1)
        dataset = dataset.map(_preprocess, num_parallel_calls=5)
        dataset = dataset.batch(250).prefetch(tf.data.experimental.AUTOTUNE)
        try:
            for _ in dataset:
                print(f"{len(corrupted_files)} corrupted files until now. Verifying file {file_name} ...",
                      end="\r")
        except:
            corrupted_files.append(file_name)

print("Corrupted files:")
print(corrupted_files)
