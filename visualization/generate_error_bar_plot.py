import os

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

from main import ROOT_DIR


def plot(x_ticks, averages, standard_deviations, figure_file_name,
         x_label="", y_label="Acurácia", title="", x_ticks_rotation=None,
         fig_size=None, font_size=20, bold_max_value=True):
    x = np.arange(len(x_ticks))

    fig, ax = plt.subplots() if fig_size is None else plt.subplots(figsize=fig_size)
    ax.set_title(title)
    ax.bar(x, averages, yerr=standard_deviations, align="center", alpha=0.9, capsize=10)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_ticks_rotation is not None:
        plt.xticks(rotation=x_ticks_rotation, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)

    if bold_max_value:
        tick_max_value = ax.xaxis.get_major_ticks()[np.argmax(averages)]
        tick_max_value.label1.set_fontweight("bold")

    for p_idx, p in enumerate(ax.patches):
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        margin = standard_deviations[p_idx] + 0.04
        position = (x + width / 2, y + height - margin)
        ax.annotate(f"{averages[p_idx]:.0%}", position, ha="center")

    if font_size is not None:
        font = {
            "size": font_size
        }
        rc("font", **font)

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "results", figure_file_name))
    plt.show()
