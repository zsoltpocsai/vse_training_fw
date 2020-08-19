"""For training the CNN model.

Can be run from the command line or used as a module by calling the 
`run` method.
"""

import argparse, os, math, configparser
import tensorflow as tf

from loss.triplet import TripletSemiHardLoss
from utils import data_generator, data_map, log, config


def run(model_path,
        training_dir=None,
        validation_dir=None,
        batch_size=30,
        group_size=10,
        epochs=10,
        learning_rate=0.05,
        margin=1.0,
        input_shape=(220, 220, 1),
        save_name=None,
        log_dir=None,
        save=True,
        verbose=1):
    """Runs the training process.

    Args:
        model_path: Full path of the CNN model.
        training_dir: Path of the dataset root directory on which the 
            training will be executed.
        validation_dir: Path of the dataset root directory which will
            be used for evaluating the model at every epoch.
        batch_size: Integer. The size of the batch produced by the 
            data generator every step.
        group_size: Integer. The size of the group produced by the 
            data generator every step. To see what a group is, check 
            out the `utils.data_generator` module.
        epochs: Integer.
        learning_rate: Float.
        margin: Float. Parameter of the triplet loss function.
        input_shape: Tuple. The shape of the model's input layer.
        save_name: The name on which the model after training will be 
            saved. If `None` the same name will be used as in `model_path`.
        log_dir: Path of the directory where the loss values will be saved.
            If `None`, the metrics won't be saved. Default to `None`.
        save: Wheather the model should be saved after training or not.
        verbose: If set to 1, prints some info to the command line.
            At 0, it will keep quiet. Default is 1.
    """

    # Loading the model

    model = tf.keras.models.load_model(model_path, compile=False)

    if verbose == 1: 
        print("Model has been successfully loaded.")

    # Compile

    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(learning_rate),
        loss=TripletSemiHardLoss(margin)
    )

    if verbose == 1: 
        print("Model has been compiled.")

    # Create the data generators
    # and calculate the steps

    train_labels_file = os.path.join(training_dir, "labels.csv")
    train_data = data_map.read_csv(train_labels_file, image_full_path=True)
    train_data = data_generator.enumerate_labels(train_data)

    train_data_gen = data_generator.for_train(
        train_data, batch_size, group_size, input_shape)
    train_steps = math.ceil(len(train_data[0]) / batch_size)

    val_labels_file = os.path.join(validation_dir, "labels.csv")
    val_data = data_map.read_csv(val_labels_file, image_full_path=True)
    val_data = data_generator.enumerate_labels(val_data)
    
    val_data_gen = data_generator.for_validation(
        val_data, batch_size, group_size, input_shape)
    val_steps = math.ceil(len(val_data[0]) / batch_size) * 2

    # Train the model

    history = model.fit_generator(
        train_data_gen,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=val_data_gen,
        validation_steps=val_steps
    )

    # Save metrics

    if log_dir != None:
        log.history(log_dir, history, append=True)

    # Save the trained model

    if save_name != None:
        model_path = os.path.join(os.path.split(model_path)[0], save_name)

    if save:
        print(f"Saving to '{model_path}'...")
        model.save(model_path, overwrite=True)

    if verbose == 1:
        print("Done!");


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_config_file")
    args = parser.parse_args()

    kwargs = config.get_train_args(args.train_config_file)

    run(**kwargs)
    