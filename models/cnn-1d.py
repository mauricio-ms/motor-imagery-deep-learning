import os
from main import ROOT_DIR
import numpy as np
from datasets import physionet
from tensorflow import keras

PHYSIONET_DIR = os.path.join(ROOT_DIR, "data/physionet")
PREPROCESSED_CSV_FILES_DIR = os.path.join(PHYSIONET_DIR, "preprocessed-csv-files")


def get_random_subject(n_subjects):
    return f"S{str(np.random.randint(1, n_subjects + 1)).zfill(3)}"


# TODO MOVER PARA PHYSIONET SCRIPT
subjects = np.array(sorted(os.listdir(PREPROCESSED_CSV_FILES_DIR)))
#n_subjects = len(subjects)
#validation_user = get_random_subject(n_subjects)
#test_user = get_random_subject(n_subjects)
#if validation_user == test_user:
#    raise Exception("The validation user can't be equals to the test user")

#train_users = filter(lambda subject: subject not in [test_user, validation_user], subjects)
train_subjects_mask = np.random.rand(len(subjects)) < 0.75
train_subjects = subjects[train_subjects_mask]
validation_subjects = subjects[~train_subjects_mask]

print(f"Training subjects: {len(train_subjects)}")
print(f"Validation subjects: {len(validation_subjects)}")

train_set = physionet.load_data(train_subjects)
validation_set = physionet.load_data(validation_subjects)
#test_set = physionet.load_data([test_user])

# print("Train Set")
# for X_batch, y_batch in train_set.take(2):
#     print(f"{X_batch.shape} - {y_batch.shape}")
#
# print("Validation Set")
# for X_batch, y_batch in validation_set.take(2):
#     print(f"{X_batch.shape} - {y_batch.shape}")

model = keras.models.Sequential([
    keras.layers.Dense(300, activation="relu",
                       kernel_initializer="he_normal",
                       input_shape=[64]),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(100, activation="relu",
                       kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(5, activation="softmax",
                       kernel_initializer="he_normal")
])
# ver qual inicialização de pesos está usando para cada camada kernel_initializer="he_normal",

optimizer = keras.optimizers.Adam(lr=1e-4)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
model.fit(train_set, epochs=10, validation_data=validation_set)

#Epoch 10/10
#2041379/2041379 [==============================] - 2516s 1ms/step - loss: 1.3163 - accuracy: 0.5362 - val_loss: 1.4643 - val_accuracy: 0.5319
