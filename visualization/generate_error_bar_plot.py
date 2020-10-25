import os

import matplotlib.pyplot as plt
import numpy as np

from main import ROOT_DIR


def plot(x_tick_labels, averages, standard_deviations, figure_file_name,
         x_label="", y_label="Acurácia", title="", figsize=None, bold_max_value=True):
    x = np.arange(len(x_tick_labels))

    fig, ax = plt.subplots() if figsize is None else plt.subplots(figsize=figsize)
    ax.bar(x, averages, yerr=standard_deviations, align="center", alpha=0.9, capsize=10)

    plt.xlabel(x_label)
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels)
    if bold_max_value:
        tick_max_value = ax.xaxis.get_major_ticks()[np.argmax(averages)]
        tick_max_value.label1.set_fontweight("bold")
    plt.ylabel(y_label)

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "results", figure_file_name))
    plt.show()
