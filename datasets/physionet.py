import os

from main import ROOT_DIR


def get_config(window_length=480, n_channels=64):
    return {
        "dir": os.path.join(ROOT_DIR, "data", "physionet", "normalized-by-sample",
                            f"window-{window_length}", "eyes-closed-left-fist-right-fist-both-fists-both-feet"),
        "window_length": window_length,
        "n_channels": n_channels
    }
