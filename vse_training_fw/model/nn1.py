"""Implementation of the NN1 model from FaceNet.

This model is described in the FaceNet paper as the NN1 model.
It is based on the Zeiler&Fergus model with added 1x1 convolutions.

References:
    'FaceNet: A Unified Embedding for Face Recognition and Clustering'
    https://arxiv.org/abs/1503.03832
"""

import sys, os, argparse
import tensorflow as tf
import save as save

sys.path.append(os.getcwd())

import utils.train_config as config
import utils.save_model as save_model


def build(input_shape):
    """Creates the model.
    
    Args:
        input_shape: A tuple representing the dimensions of the input
            image in the format of '(width, height, channels)'.

    Returns:
        A tf.keras.Model object.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 7, activation="relu", input_shape=input_shape, strides=2, padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 1, activation="relu"),
        tf.keras.layers.Conv2D(192, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same"),
        tf.keras.layers.Conv2D(192, 1, activation="relu"),
        tf.keras.layers.Conv2D(384, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same"),
        tf.keras.layers.Conv2D(384, 1, activation="relu"),
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(256, 1, activation="relu"),
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(256, 1, activation="relu"),
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.Dense(128, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

    return model


if __name__ == "__main__":
    """Builds the model and saves it.
    
    In the argument the path of the training parameters config file
    containing the model input shape configuration must be given.

    It saves the model to the root 'models' directory defined in the
    'config.ini' file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("run_config_file")
    args = parser.parse_args()

    save_name = "nn1"
    input_shape = config.read(args.run_config_file)["input_shape"]
    model = build(input_shape)

    save_model.save(model, save_name)
