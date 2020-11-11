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

MODEL_DIR = os.path.join(ROOT_DIR, "models-weights", "physionet", "all-classes")

print(f"{time.asctime()} - Starting evaluation model ...")

config = get_config()
filenames = datasets.get_filenames(config)

n_classes = 5
cm = np.zeros(shape=(n_classes, n_classes), dtype=np.int64)

n_splits = 10
kf = KFold(n_splits=n_splits)
for fold, (train_index, test_index) in enumerate(kf.split(filenames)):
    print("\nFold: ", fold)
    test_filenames = filenames[test_index]

    print("Test filenames:")
    print(test_filenames)

    model_name = f"independent-{n_splits}-cross-validation-fold-{fold}"
    model_weights_filepath = os.path.join(MODEL_DIR, f"{model_name}.h5")

    model = load_model(model_weights_filepath, output_classes=n_classes)

    X, y = datasets.load(config, test_filenames, batch_size=1, expand_dim=False, xy_format=True)
    y_predicted = model.predict(X)
    cm = cm + metrics.confusion_matrix(y, y_predicted.argmax(axis=1))

    tf.keras.backend.clear_session()

print(f"{time.asctime()} - Evaluation model end!")

classes = ["Olhos Fechados", "Punho Esquerdo", "Punho Direito", "Punhos", "Pés"]
figure_filepath = os.path.join(ROOT_DIR, "results", "physionet", "confusion-matrix-all-classes.png")

plot_confusion_matrix(cm, classes, figure_filepath)
