import os
import re
import time

import numpy as np
import tensorflow as tf
from sklearn import metrics

import datasets.bciiviia as dataset
from main import ROOT_DIR
from models.bciiviia.cnn1d import load_model
from visualization.confusion_matrix_plot_helper import plot_confusion_matrix

MODEL_DIR = os.path.join(ROOT_DIR, "models-weights", "bci-iv-iia", "multiple-classes")

print(f"{time.asctime()} - Starting evaluation model ...")

filenames = dataset.get_filenames()

n_classes = 4
cm = np.zeros(shape=(n_classes, n_classes), dtype=np.int64)

n_subjects = 9
subjects = np.array(range(n_subjects)) + 1
for test_subject in subjects:
    print("\nTest subject: ", test_subject)

    test_filenames = list(filter(lambda f: re.match(f"A0{test_subject}([TE]).tfrecord", f), filenames))

    print("Test filenames:")
    print(test_filenames)

    model_name = f"test-subject-{test_subject}"
    model_weights_filepath = os.path.join(MODEL_DIR, f"{model_name}.h5")
    model = load_model(model_weights_filepath, output_classes=n_classes)

    X, y = dataset.load_data(test_filenames, batch_size=1, expand_dim=False, xy_format=True)
    y_predicted = model.predict(X)
    cm = cm + metrics.confusion_matrix(y, y_predicted.argmax(axis=1))

    tf.keras.backend.clear_session()

print(cm)
print(f"{time.asctime()} - Evaluation model end!")

classes = ["Mão Esquerda", "Mão Direita", "Pés", "Língua"]
figure_filepath = os.path.join(ROOT_DIR, "results", "bci-iv-iia", "confusion-matrix-multiple-classes.png")

plot_confusion_matrix(cm, classes, figure_filepath)
