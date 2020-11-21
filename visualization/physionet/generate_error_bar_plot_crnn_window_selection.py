import os

from main import ROOT_DIR
from visualization.error_bar_plot_helper import plot_error_bar

x_tick_labels = ["(62.5, 31.25) ms",
                 "(0.25, 0.25) s",
                 "(0.5, 0.5) s",
                 "(1, 1) s",
                 "(2, 0) s",
                 "(3, 0) s",
                 "(4, 0) s"]

averages = [0.5662032306194306,
            0.596405291557312,
            0.6108131289482117,
            0.6309646427631378,
            0.6347777664661407,
            0.6539797961711884,
            0.6635959625244141]
standard_deviations = [0.017610818473707008,
                       0.021647917581321955,
                       0.026114269769024732,
                       0.030232735557993627,
                       0.024183306005983927,
                       0.030908795271991613,
                       0.030938747643292974]

figure_filepath = os.path.join(ROOT_DIR, "results", "physionet", "results-crnn-window-selection.png")
plot_error_bar(x_tick_labels, averages, standard_deviations, figure_filepath,
               x_label="(Janela, Deslocamento) Unidade de Tempo", fig_size=(19, 12))
