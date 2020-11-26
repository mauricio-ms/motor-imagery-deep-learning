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

averages = [0.5399867296218872,
            0.5618333339691162,
            0.583931815624237,
            0.6067323327064514,
            0.5949696958065033,
            0.5860606074333191,
            0.609090906381607]
standard_deviations = [0.012469947780813551,
                       0.019838599458321317,
                       0.020426887088374802,
                       0.03249515129893599,
                       0.027812140088209165,
                       0.035179484337870895,
                       0.04437818789499396]

figure_filepath = os.path.join(ROOT_DIR, "results", "physionet", "results-crnn-window-selection.png")
plot_error_bar(x_tick_labels, averages, standard_deviations, figure_filepath,
               x_label="(Janela, Deslocamento) Unidade de Tempo",
               fig_size=(19, 12), index_best_result=3)
