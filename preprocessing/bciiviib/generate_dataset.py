import os

import mne

import preprocessing.bciiv.generate_dataset as generate_dataset_bciiv
from main import ROOT_DIR

mne.set_log_level("WARNING")

EVENTS_ENUM = {
    "768": "NEW_TRIAL",
    "769": "LEFT_HAND",
    "770": "RIGHT_HAND",
    "783": "CUE_UNKNOWN",
    "1023": "REJECTED_TRIAL"
}

LABELS_ENUM = {
    1: "LEFT_HAND",
    2: "RIGHT_HAND"
}

BCI_IV_IIB_DIR = os.path.join(ROOT_DIR, "data/bci-iv-iib")


def generate(window_size=1000, classes=None):
    if classes is None:
        classes = ["LEFT_HAND", "RIGHT_HAND"]

    label_value = 0
    labels = {}
    if "LEFT_HAND" in classes:
        labels["LEFT_HAND"] = label_value
        label_value += 1
    if "RIGHT_HAND" in classes:
        labels["RIGHT_HAND"] = label_value
        label_value += 1

    generate_dataset_bciiv.generate(BCI_IV_IIB_DIR, EVENTS_ENUM, LABELS_ENUM, labels, classes, window_size)


generate(window_size=750)
