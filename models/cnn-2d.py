import os
from main import ROOT_DIR
import numpy as np
from datasets import physionet
from tensorflow import keras

PHYSIONET_DIR = os.path.join(ROOT_DIR, "data/physionet")
PREPROCESSED_CSV_FILES_DIR = os.path.join(PHYSIONET_DIR, "preprocessed-csv-files")

# TODO MOVER PARA PHYSIONET SCRIPT
subjects = np.array(sorted(os.listdir(PREPROCESSED_CSV_FILES_DIR)))
train_subjects_mask = np.random.rand(len(subjects)) < 0.75
train_subjects = subjects[train_subjects_mask]
validation_subjects = subjects[~train_subjects_mask]

print(f"Training subjects: {len(train_subjects)}")
print(f"Validation subjects: {len(validation_subjects)}")

train_set = physionet.load_data(train_subjects)
validation_set = physionet.load_data(validation_subjects)
model = keras.models.Sequential([
    keras.layers.Input(shape=(None, 64, 1)),
    keras.layers.Conv1D(32, 3, activation="relu",
                        kernel_initializer="he_normal"),
    keras.layers.Conv1D(64, 3, activation="relu",
                        kernel_initializer="he_normal"),
    keras.layers.Conv1D(128, 3, activation="relu",
                        kernel_initializer="he_normal"),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation="relu",
                       kernel_initializer="he_normal"),
    keras.layers.Dense(5, activation="softmax")
])

optimizer = keras.optimizers.Adam(lr=1e-4)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
model.fit(train_set, epochs=10, validation_data=validation_set)
