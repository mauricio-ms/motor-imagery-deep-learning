from tensorflow import keras


def load_model(weights_filepath, output_classes=2, window_length=480):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(window_length, 64)),
        keras.layers.Conv1D(8, 3, activation="elu", padding="SAME"),
        keras.layers.AveragePooling1D(pool_size=3),
        keras.layers.Conv1D(16, 3, activation="elu", padding="SAME"),
        keras.layers.AveragePooling1D(pool_size=3),
        keras.layers.Conv1D(32, 3, activation="elu", padding="SAME"),
        keras.layers.AveragePooling1D(pool_size=3),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="elu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_classes, activation="softmax")
    ])

    model.load_weights(weights_filepath)
    optimizer = keras.optimizers.Adam(lr=1e-4)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])

    return model
