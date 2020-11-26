import os

from main import ROOT_DIR
from visualization.error_bar_plot_helper import plot_error_bar

x_tick_labels = ["(1, 0) s",
                 "(1, 0.5) s",
                 "(1, 1) s",
                 "(2, 0) s",
                 "(2, 1) s",
                 "(2, 2) s",
                 "(3, 0) s",
                 "(4, 0) s"]

averages = [0.7893333315849305,
            0.6398961007595062,
            0.625550502538681,
            0.7974343478679657,
            0.6765791237354278,
            0.6955555498600006,
            0.804909098148346,
            0.8134343445301055]
standard_deviations = [0.03320707963204236,
                       0.026240448023486868,
                       0.018550820517630115,
                       0.04090089593933209,
                       0.029132393758999547,
                       0.028864336218454776,
                       0.04230401519084769,
                       0.030207012556788274]

figure_filepath = os.path.join(ROOT_DIR, "results", "physionet", "results-cnn2d-window-selection.png")
plot_error_bar(x_tick_labels, averages, standard_deviations, figure_filepath,
               x_label="(Janela, Deslocamento) Unidade de Tempo",
               fig_size=(19, 12), index_best_result=6)
