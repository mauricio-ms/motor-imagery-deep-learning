from tensorflow import keras


def load_model(weights_filepath, output_classes=2, window_length=480):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(window_length, 64)),
        keras.layers.Conv1D(32, 10, activation="elu", padding="SAME",
                            kernel_constraint=keras.constraints.max_norm(0.25)),
        keras.layers.AveragePooling1D(pool_size=3),
        keras.layers.Dropout(0.5),
        keras.layers.Conv1D(64, 10, activation="elu", padding="SAME",
                            kernel_constraint=keras.constraints.max_norm(0.25)),
        keras.layers.AveragePooling1D(pool_size=3),
        keras.layers.Dropout(0.5),
        keras.layers.Conv1D(128, 10, activation="elu", padding="SAME",
                            kernel_constraint=keras.constraints.max_norm(0.25)),
        keras.layers.AveragePooling1D(pool_size=3),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation="elu", kernel_constraint=keras.constraints.max_norm(0.25)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_classes, activation="softmax")
    ])

    model.load_weights(weights_filepath)
    optimizer = keras.optimizers.Adam(lr=1e-4)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])

    return model
