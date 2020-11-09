import os

from main import ROOT_DIR
from visualization.error_bar_plot_helper import plot_error_bar

x_tick_labels = ["3, 3, 3",
                 "5, 5, 5",
                 "7, 7, 7",
                 "7, 5, 3"]

averages = [0.8231717228889466,
            0.8235555589199066,
            0.8213131308555603,
            0.8203838467597961]
standard_deviations = [0.03842985128314884,
                       0.03896143993477434,
                       0.03627848990607666,
                       0.03903387884648613]

figure_filepath = os.path.join(ROOT_DIR, "results", "physionet", "results-conv2d-kernel-size-selection.png")
plot_error_bar(x_tick_labels, averages, standard_deviations, figure_filepath,
               x_label="Kernel 1º camada, Kernel 2º camada, Kernel 3º camada")
