import os

from main import ROOT_DIR
from visualization.error_bar_plot_helper import plot_error_bar

x_tick_labels = ["Todas Sessões",
                 "Sessões Sem Feedback",
                 "Sessões Com Feedback"]

averages = [0.7859424220191108,
            0.7517999476856656,
            0.8414837982919481]
standard_deviations = [0.05132932342000875,
                       0.043330507916977644,
                       0.0678091871746606]

figure_filepath = os.path.join(ROOT_DIR, "results", "bci-iv-iib", "results-by-sessions.png")
plot_error_bar(x_tick_labels, averages, standard_deviations, figure_filepath, fig_size=(12, 9))
