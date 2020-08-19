"""Saves tf.keras models."""

import tensorflow as tf
import configparser, os


def save(model, save_name):
    """Saves the model with some additional info.
    
    The save root directory is the one configured in the 'config.ini' file.
    Also creates an additional .txt file right next to the model, which
    contains the summary information of the model.
    
    Args:
        model: A tf.keras.Model object.
        save_name: A name in which the model will be saved. Can be a path
            in which case subfolders will be automatically created.
    """
    dirs = configparser.ConfigParser()
    dirs.read("config/dir_config.ini")

    save_name = os.path.splitext(save_name)[0]
    path = os.path.join(dirs["save_dirs"]["models"], save_name + ".h5")
    info = os.path.join(dirs["save_dirs"]["models"], save_name + "_info.txt")

    with open(info, "w") as file:
        model.summary(print_fn=lambda x: file.write(f"{x}\n"))
    model.save(path, overwrite=False)
    