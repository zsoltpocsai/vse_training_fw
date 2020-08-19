"""Implementation of the Inception architecture.

This is the variation described in the FaceNet paper, which is a
slightly modified version of the GoogLeNet model.

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


def inception_modul1(x, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pool_proj, l2=False):
    # 1x1
    x1 = tf.keras.layers.Conv2D(f_1x1, 1, activation="relu", padding="same") (x)

    # 3x3
    x2 = tf.keras.layers.Conv2D(f_3x3_reduce, 1, activation="relu", padding="same") (x)
    x2 = tf.keras.layers.Conv2D(f_3x3, 3, activation="relu", padding="same") (x2)

    # 5x5
    x3 = tf.keras.layers.Conv2D(f_5x5_reduce, 1, activation="relu", padding="same") (x)
    x3 = tf.keras.layers.Conv2D(f_5x5, 5, activation="relu", padding="same") (x3)

    # pool
    if l2:
        x4 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=3)) (x)
    else:
        x4 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=1, padding="same") (x)

    x4 = tf.keras.layers.Conv2D(f_pool_proj, 1, activation="relu", padding="same") (x4)

    out = tf.keras.layers.Concatenate() ([x1, x2, x3, x4])
    return out


def inception_modul2(x, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5):
    # 3x3
    x1 = tf.keras.layers.Conv2D(f_3x3_reduce, 1, activation="relu", padding="same") (x)
    x1 = tf.keras.layers.Conv2D(f_3x3, 3, activation="relu", padding="same", strides=2) (x1)

    # 5x5
    x2 = tf.keras.layers.Conv2D(f_5x5_reduce, 1, activation="relu", padding="same") (x)
    x2 = tf.keras.layers.Conv2D(f_5x5, 5, activation="relu", padding="same", strides=2) (x2)

    # pool
    x3 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same") (x)

    out = tf.keras.layers.Concatenate() ([x1, x2, x3])
    return out


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
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same") (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Conv2D(64, 1, activation="relu", padding="same") (x)
    x = tf.keras.layers.Conv2D(192, 3, activation="relu", padding="same") (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same") (x)

    x = inception_modul1(x, 64, 96, 128, 16, 32, 32)
    x = inception_modul1(x, 64, 96, 128, 32, 64, 64, l2=True)
    x = inception_modul2(x, 128, 256, 32, 64)

    x = inception_modul1(x, 256, 96, 192, 32, 64, 128, l2=True)
    x = inception_modul1(x, 224, 112, 224, 32, 64, 128, l2=True)
    x = inception_modul1(x, 192, 128, 256, 32, 64, 128, l2=True)
    x = inception_modul1(x, 160, 144, 288, 32, 64, 128, l2=True)
    x = inception_modul2(x, 160, 256, 64, 128)

    x = inception_modul1(x, 384, 192, 384, 48, 128, 128, l2=True)
    x = inception_modul1(x, 384, 192, 384, 48, 128, 128)

    x = tf.keras.layers.GlobalAveragePooling2D() (x)
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

    save_name = "nn2"
    input_shape = config.read(args.run_config_file)["input_shape"]
    model = build(input_shape)

    save_model.save(model, save_name)
