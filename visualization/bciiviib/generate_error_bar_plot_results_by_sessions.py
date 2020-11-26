import os

from main import ROOT_DIR
from visualization.error_bar_plot_helper import plot_error_bar

x_tick_labels = ["Todas Sessões",
                 "Sessões Sem Feedback",
                 "Sessões Com Feedback"]

averages = [0.7113788723945618,
            0.6668007440037198,
            0.7414302892155118]
standard_deviations = [0.05627800388053611,
                       0.029368594488640088,
                       0.1054164690746184]

figure_filepath = os.path.join(ROOT_DIR, "results", "bci-iv-iib", "results-by-sessions.png")
plot_error_bar(x_tick_labels, averages, standard_deviations, figure_filepath, fig_size=(12, 9))
