"""Evaluates the CNN model with FAR/VAL metrics.

Can be run from the command line or used as a module by calling the 
`run` method.
"""

import argparse, os, math, csv
import tensorflow as tf
import numpy as np

from utils import data_generator, data_map, config, log
from metrics.valrate import ValidationRate
from loss.triplet import TripletSemiHardLoss


def _generate_tresholds(n, scalar, offset, divisor):
    tresholds = [(offset + i * scalar) / divisor for i in range(n)]
    return tresholds


def run(model_path, 
        dataset_dir,
        input_shape=(220, 220, 1),
        batch_size=60,
        group_size=6,
        log_dir=None,
        verbose=1):
    """Runs the evaluation process.

    Evaluates the CNN model with the VAL/FAR metrics. Treshold values
    can be changed by altering the parameters passed to the 
    `_generate_tresholds` method.

    Args:
        model_path: Full path of the CNN model.
        dataset_dir: Path of the dataset root directory on which the 
            evaluation will be executed.
        input_shape: Tuple. The shape of the model's input layer.
        batch_size: Integer. The size of the batch produced by the 
            data generator every step.
        group_size: Integer. The size of the group produced by the 
            data generator every step. To see what a group is, check 
            out the `utils.data_generator` module.
        log_dir: Path of the directory where the measured values will
            be saved.
        verbose: If set to 1, prints some info to the command line.
            At 0, it will keep quiet. Default is 1.
    """

    # Loads the model

    model = tf.keras.models.load_model(model_path, compile=False)

    if verbose == 1:
        print("Model has been successfully loaded.")

    # Prepares the data
    # and calculates the steps

    labels_file = os.path.join(dataset_dir, "labels.csv")

    eval_data = data_map.read_csv(labels_file, image_full_path=True)
    eval_data = data_generator.enumerate_labels(eval_data)

    steps = math.ceil(len(eval_data[0]) / batch_size)
    steps *= 2 # running through the data twice gives more accurate results

    # Sets up the treshold values
    # Arguments can be changed for a better cover of treshold values
    tresholds = _generate_tresholds(15, 32, 8, 10000)

    for t in tresholds:

        # Compiles

        val_far_metrics = [
            ValidationRate(treshold=t),
            ValidationRate(name="far", treshold=t, false_accepts=True)
        ]

        model.compile(
            loss=TripletSemiHardLoss(),
            metrics=[val_far_metrics])

        # Creates the generator

        generator = data_generator.for_validation(eval_data, batch_size, group_size, 
            input_shape)

        # Evaluates with VAL and FAR

        scalars = model.evaluate_generator(
            generator,
            steps,
            verbose=1)

        # Saves metrics

        if log_dir != None:
            log.val_far_evaluation(log_dir, scalars, model.metrics_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_config_file")
    args = parser.parse_args()

    kwargs = config.get_evaluate_args(args.run_config_file)

    run(**kwargs)
