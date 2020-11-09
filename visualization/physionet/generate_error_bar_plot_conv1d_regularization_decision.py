import os

from main import ROOT_DIR
from visualization.error_bar_plot_helper import plot_error_bar

x_tick_labels = ["Base",
                 "Experimento 1",
                 "Experimento 2",
                 "Experimento 3"]

averages = [0.830767673254013,
            0.8382424175739288,
            0.8395757555961609,
            0.843676769733429]
standard_deviations = [0.03555372341872865,
                       0.03475793447084474,
                       0.03761847172151291,
                       0.03379597766448776]

figure_filepath = os.path.join(ROOT_DIR, "results", "physionet", "results-conv1d-regularization-decision.png")
plot_error_bar(x_tick_labels, averages, standard_deviations, figure_filepath)
