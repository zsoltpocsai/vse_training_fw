"""Functions for saving metrics."""

import tensorflow as tf
import os, csv, math


def val_far_evaluation(log_dir, scalars, metrics_names):
    """Saves the VAL and FAR metrics into a csv file.
    
    Args:
        log_dir: The directory where the log file will be stored.
        scalars: A list of scalars. The output of the 
            `tf.keras.Model.evaluate_generator` method.
        metrics_name: A list containing the scalar display labels. This
            is the atttribute `tf.keras.Model.metrics_names`.
    """
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    i = metrics_names.index("loss")
    scalars.pop(i)
    metrics_names.pop(i)

    file_path = os.path.join(
        log_dir,
        "val_far.csv"
    )

    with open(file_path, "a", newline="") as file:

        writer = csv.writer(file)

        # if it is opened first time, write a header
        if os.path.getsize(file_path) == 0:
            header = [metrics_names[0], metrics_names[1]]
            writer.writerow(header)

        row = [
            math.floor(scalars[0] * 10000) / 10000, 
            math.floor(scalars[1] * 10000) / 10000
        ]

        writer.writerow(row)


def history(log_dir, history, append=False):
    """Saves the training loss values into a csv file.
    
    Args:
        log_dir: The directory where the log file will be stored.
        history: A history object. The output of the 
            `tf.keras.Model.fit_generator` method.
        append: If `True` the existing file will be opened and new data
            will be appended at the end, otherwise the file will be
            overwritten. Default is `False`.
    """
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    history_dic = history.history

    train_loss = [value for value in history_dic["loss"]]
    val_loss = [value for value in history_dic["val_loss"]]

    losses = zip(train_loss, val_loss)

    if append:
        mode = "a"
    else:
        mode = "w"

    file_path = os.path.join(log_dir, "loss.csv")

    with open(file_path, mode, newline="") as file:

        writer = csv.writer(file)

        # if it is opened first time, write a header
        if os.path.getsize(file_path) == 0:
            header = ["train_loss", "val_loss"]
            writer.writerow(header)

        for train_loss, val_loss in losses:
            row = [
                math.floor(train_loss * 10000) / 10000, 
                math.floor(val_loss * 10000) / 10000
            ]
            writer.writerow(row)
