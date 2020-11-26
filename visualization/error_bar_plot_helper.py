import matplotlib.pyplot as plt
import numpy as np


def plot_error_bar(x_ticks, averages, standard_deviations, figure_filepath,
                   x_label="", y_label="Acurácia", title="", x_ticks_rotation=None,
                   fig_size=(10, 7), font_size=18, bold_max_value=True, index_best_result=None):
    x = np.arange(len(x_ticks))

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title(title)
    ax.bar(x, averages, yerr=standard_deviations, align="center", alpha=0.9, capsize=10)

    label_pad = 15
    plt.xlabel(x_label, labelpad=label_pad, fontsize=font_size, weight="bold")
    plt.ylabel(y_label, labelpad=label_pad, fontsize=font_size, weight="bold")
    ax.tick_params(axis="y", labelsize=font_size)

    if x_ticks_rotation is not None:
        plt.xticks(rotation=x_ticks_rotation, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks, fontsize=font_size)

    index_best_result = np.argmax(averages) if index_best_result is None else index_best_result
    if bold_max_value:
        tick_max_value = ax.xaxis.get_major_ticks()[index_best_result]
        tick_max_value.label1.set_fontweight("bold")

    for p_idx, p in enumerate(ax.patches):
        width = p.get_width()
        x = p.get_x()
        position = (x + width / 2, 0.05)
        weight = "bold" if p_idx == index_best_result else None
        ax.annotate(f"{averages[p_idx]:.0%}", position, ha="center",
                    fontsize=font_size, weight=weight)

    plt.tight_layout()
    plt.savefig(figure_filepath)
    plt.show()
