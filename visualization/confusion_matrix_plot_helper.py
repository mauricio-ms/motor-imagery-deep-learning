import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


def plot_confusion_matrix(cm, classes, figure_filepath):
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(14, 7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, fmt="g", cmap="Blues", annot_kws={"size": 16}, cbar=False)
    plt.xlabel("Predição", fontweight="bold")
    plt.ylabel("Atual", fontweight="bold")

    plt.tight_layout()
    plt.savefig(figure_filepath)
    plt.show()
