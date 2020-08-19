"""Implementation of the 18-layer ResNet architecture.

Reference:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
"""

import sys, os, argparse
import tensorflow as tf

sys.path.append(os.getcwd())
from utils import save_model, config


def build(input_shape):
    """Creates the model.
    
    Args:
        input_shape: A tuple representing the dimensions of the input
            image in the format of '(width, height, channels)'.

    Returns:
        A tf.keras.Model object.
    """
    input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 7, activation="relu", padding="same", strides=2) (input)
    output = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same") (x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same") (output)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Conv2D(64, 3, activation=None, padding="same") (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Add() ([x, output])
    output = tf.keras.layers.Activation("relu") (x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same") (output)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Conv2D(64, 3, activation=None, padding="same") (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Add() ([x, output])
    output = tf.keras.layers.Activation("relu") (x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", strides=2) (output)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Conv2D(128, 3, activation=None, padding="same") (x)
    x = tf.keras.layers.BatchNormalization() (x)
    output = tf.keras.layers.Conv2D(128, 1, activation="relu", padding="same", strides=2) (output)
    x = tf.keras.layers.Add() ([x, output])
    output = tf.keras.layers.Activation("relu") (x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same") (output)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Conv2D(128, 3, activation=None, padding="same") (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Add() ([x, output])
    output = tf.keras.layers.Activation("relu") (x)

    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", strides=2) (output)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Conv2D(256, 3, activation=None, padding="same") (x)
    x = tf.keras.layers.BatchNormalization() (x)
    output = tf.keras.layers.Conv2D(256, 1, activation="relu", padding="same", strides=2) (output)
    x = tf.keras.layers.Add() ([x, output])
    output = tf.keras.layers.Activation("relu") (x)

    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same") (output)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Conv2D(256, 3, activation=None, padding="same") (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Add() ([x, output])
    output = tf.keras.layers.Activation("relu") (x)

    x = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same", strides=2) (output)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Conv2D(512, 3, activation=None, padding="same") (x)
    x = tf.keras.layers.BatchNormalization() (x)
    output = tf.keras.layers.Conv2D(512, 1, activation="relu", padding="same", strides=2) (output)
    x = tf.keras.layers.Add() ([x, output])
    output = tf.keras.layers.Activation("relu") (x)

    x = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same") (output)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Conv2D(512, 3, activation=None, padding="same") (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Add() ([x, output])
    output = tf.keras.layers.Activation("relu") (x)

    x = tf.keras.layers.GlobalMaxPool2D() (output)
    x = tf.keras.layers.Dense(128, activation=None) (x)
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) (x)

    return tf.keras.Model(input, output)


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

    model_config = config.read_run_config(args.run_config_file)
    save_name = model_config["model_name"]
    input_shape = model_config["input_shape"]
    model = build(input_shape)

    save_model.save(model, save_name)
