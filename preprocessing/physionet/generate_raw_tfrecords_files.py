import os
import re
import helpers.file_system_helper as fsh
from preprocessing.physionet.config import PHYSIONET_DIR, DAMAGED_SUBJECTS
from preprocessing.physionet.EdfFile import EdfFile
import tensorflow as tf
from tensorflow.core.example.feature_pb2 import Features, Feature, FloatList
from tensorflow.core.example.example_pb2 import Example

RAW_EDF_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-edf-files")
RAW_TFRECORD_FILES_DIR = os.path.join(PHYSIONET_DIR, "raw-tfrecord-files")

fsh.recreate_dir(RAW_TFRECORD_FILES_DIR)

subjects = filter(lambda s: s not in DAMAGED_SUBJECTS, sorted(os.listdir(RAW_EDF_FILES_DIR)))
for subject in filter(lambda f: re.match("S(\\d+)", f), subjects):
    tfrecord_subject_path_dir = os.path.join(RAW_TFRECORD_FILES_DIR, subject)
    fsh.mkdir_if_not_exists(tfrecord_subject_path_dir)

    edf_subject_path_dir = os.path.join(RAW_EDF_FILES_DIR, subject)
    edf_file_names = sorted(os.listdir(edf_subject_path_dir))
    for edf_file_name in filter(lambda f: f.endswith(".edf"), edf_file_names):
        print(f"Generating TFRecord file from the file {edf_file_name} ...", end="\r")
        edf_file = EdfFile(edf_subject_path_dir, edf_file_name)

        tfrecord_path_file = edf_file.get_path_file(tfrecord_subject_path_dir, "tfrecord")
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(tfrecord_path_file, options) as tfr:
            for n_sample in range(edf_file.n_samples):
                eeg_record = edf_file.data[n_sample, :-1]
                labels = edf_file.data[n_sample, -1]
                eeg_example = Example(
                    features=Features(
                        feature={
                            "X": Feature(float_list=FloatList(value=eeg_record)),
                            "y": Feature(float_list=FloatList(value=[labels]))
                        }
                    )
                )
                tfr.write(eeg_example.SerializeToString())
