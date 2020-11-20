import os

from main import ROOT_DIR


def get_config(classes, window_length=480, n_channels=64):
    classes_dirname = "-".join(classes)
    return {
        "classes_dirname": classes_dirname,
        "dir": os.path.join(ROOT_DIR, "data", "physionet", "normalized-by-sample",
                            f"window-{window_length}", classes_dirname),
        "window_length": window_length,
        "n_channels": n_channels
    }
