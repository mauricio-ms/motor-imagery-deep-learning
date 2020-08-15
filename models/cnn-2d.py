from datasets import physionet
from tensorflow import keras

train_set, validation_set = physionet.load_data(convert_to_2d=True, expand_dim=True)

model = keras.models.Sequential([
    keras.layers.Conv1D(32, 3, activation="relu",
                        kernel_initializer="he_normal", padding="SAME",
                        input_shape=[10, 11, 1]),
    keras.layers.Conv1D(64, 3, activation="relu",
                        kernel_initializer="he_normal", padding="SAME"),
    keras.layers.Conv1D(128, 3, activation="relu",
                        kernel_initializer="he_normal", padding="SAME"),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024, activation="relu",
                       kernel_initializer="he_normal"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5, activation="softmax")
])

optimizer = keras.optimizers.Adam(lr=1e-4)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
model.fit(train_set, epochs=10, validation_data=validation_set)
