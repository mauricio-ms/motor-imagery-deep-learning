from visualization.generate_error_bar_plot import plot

x_tick_labels = ["15",
                 "25",
                 "50",
                 "75",
                 "100"]

averages = [0.8360201954841614,
            0.843676769733429,
            0.8349090933799743,
            0.8333131372928619,
            0.837818193435669]
standard_deviations = [0.03176410952404728,
                       0.03379597766448776,
                       0.03598271854808842,
                       0.0338910820377901,
                       0.03344892986830442]

plot(x_tick_labels, averages, standard_deviations,
     "results-conv1d-maxnorm-hyperparameter-selection.png",
     x_label="Hiper-parâmetro Max-Norm")
