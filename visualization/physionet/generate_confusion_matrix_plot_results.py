import os
import time

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import KFold

import datasets.datasets as datasets
from datasets.physionet import get_config
from main import ROOT_DIR
from models.physionet.cnn1d import load_model
from visualization.confusion_matrix_plot_helper import plot_confusion_matrix

PHYSIONET_MODELS_DIR = os.path.join(ROOT_DIR, "models-weights", "physionet")
LABELS_ENUM = {
    "eyes-closed": "Olhos Fechados",
    "left-fist": "Punho Esquerdo",
    "right-fist": "Punho Direito",
    "both-fists": "Punhos",
    "both-feet": "Pés"
}


def plot_confusion_matrix_for_model(classes):
    config = get_config(classes)
    model_dirname = config["classes_dirname"]
    print(f"{time.asctime()} - Starting confusion matrix plot generation for model {model_dirname} ...")
    model_dir = os.path.join(PHYSIONET_MODELS_DIR, model_dirname)

    filenames = datasets.get_filenames(config)

    n_classes = len(classes)
    cm = np.zeros(shape=(n_classes, n_classes), dtype=np.int64)

    n_splits = 10
    kf = KFold(n_splits=n_splits)
    for fold, (train_index, test_index) in enumerate(kf.split(filenames)):
        print("\nFold: ", fold)
        test_filenames = filenames[test_index]

        print("Test filenames:")
        print(test_filenames)

        model_filename = f"independent-{n_splits}-cross-validation-fold-{fold}"
        model_weights_filepath = os.path.join(model_dir, f"{model_filename}.h5")

        model = load_model(model_weights_filepath, output_classes=n_classes)

        X, y = datasets.load(config, test_filenames, batch_size=1, expand_dim=False, xy_format=True)
        y_predicted = model.predict(X)
        cm = cm + metrics.confusion_matrix(y, y_predicted.argmax(axis=1))

        tf.keras.backend.clear_session()

    print(f"{time.asctime()} - Evaluation model end!")

    figure_filepath = os.path.join(ROOT_DIR, "results", "physionet", f"confusion-matrix-{model_dirname}.png")

    classes_labels = [LABELS_ENUM[c] for c in classes]
    plot_confusion_matrix(cm, classes_labels, figure_filepath)


# To generate the confusion matrix to 5-class scenario
# plot_confusion_matrix_for_model(["eyes-closed", "left-fist", "right-fist", "both-fists", "both-feet"])

# To generate the confusion matrix to 4-class scenario
plot_confusion_matrix_for_model(["eyes-closed", "left-fist", "right-fist", "both-feet"])
