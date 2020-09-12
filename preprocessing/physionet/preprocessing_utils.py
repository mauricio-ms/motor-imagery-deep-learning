import numpy as np
import tensorflow as tf


def convert_to_2D(X):
    X_2D = np.zeros([10, 11])

    X_2D[0] = (0, 0, 0, 0, X[21], X[22], X[23], 0, 0, 0, 0)
    X_2D[1] = (0, 0, 0, X[24], X[25], X[26], X[27], X[28], 0, 0, 0)
    X_2D[2] = (0, X[29], X[30], X[31], X[32], X[33], X[34], X[35], X[36], X[37], 0)
    X_2D[3] = (0, X[38], X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[39], 0)
    X_2D[4] = (X[42], X[40], X[7], X[8], X[9], X[10], X[11], X[12], X[13], X[41], X[43])
    X_2D[5] = (0, X[44], X[14], X[15], X[16], X[17], X[18], X[19], X[20], X[45], 0)
    X_2D[6] = (0, X[46], X[47], X[48], X[49], X[50], X[51], X[52], X[53], X[54], 0)
    X_2D[7] = (0, 0, 0, X[55], X[56], X[57], X[58], X[59], 0, 0, 0)
    X_2D[8] = (0, 0, 0, 0, X[60], X[61], X[62], 0, 0, 0, 0)
    X_2D[9] = (0, 0, 0, 0, 0, X[63], 0, 0, 0, 0, 0)

    return X_2D


@tf.function
def preprocess(serialized_eeg_record, X_shape=None, expand_dim=False):
    """
        The reason to expand_dim parameter is because TensorFlow expects a certain input shape
        for it's Deep Learning Model. For example a Convolution Neural Network expect:

        (<number of samples>, <x_dim sample>, <y_dim sample>, <number of channels>)
    """
    feature_description = {
        "X": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_eeg_records = tf.io.parse_single_example(serialized_eeg_record, feature_description)
    X = parsed_eeg_records["X"]
    X = tf.io.parse_tensor(X, out_type=tf.float64)

    if X_shape is not None:
        X.set_shape(X_shape)
    y = parsed_eeg_records["y"]

    if expand_dim:
        X = X[..., np.newaxis]

    return X, y
