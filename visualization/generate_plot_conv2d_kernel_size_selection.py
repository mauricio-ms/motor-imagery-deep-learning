from visualization.generate_error_bar_plot import plot

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

plot(x_tick_labels, averages, standard_deviations, "results-conv2d-kernel-size-selection.png",
     x_label="Kernel 1º camada, Kernel 2º camada, Kernel 3º camada")
