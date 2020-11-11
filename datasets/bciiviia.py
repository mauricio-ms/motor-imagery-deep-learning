import os

from main import ROOT_DIR


def get_config(window_length=750, n_channels=22):
    return {
        "dir": os.path.join(ROOT_DIR, "data", "bci-iv-iia", "normalized-by-sample",
                            f"window-{window_length}", "left_hand-right_hand-both_feet-tongue"),
        "window_length": window_length,
        "n_channels": n_channels
    }
