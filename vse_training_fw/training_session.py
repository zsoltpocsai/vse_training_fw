"""For training the CNN model running cycles of training.

It gives the opportunity to run a training automatically for x epoch, 
save the model (creating snapshots) and continue the training for 
another x epoch. Also the learning rate can be altered as well using 
schedules.

Can be run from the command line or used as a module by calling the 
`run` method.
"""

import argparse, os, math
import train, embed, evaluate
import tensorflow as tf
from utils import config


def time_decay(cycle, lr0):
    decay = 1.0
    lr = lr0 / (1 + decay * cycle)
    return lr


def step_decay(cycle, lr0):
    drop = 0.5
    cycle_drop = 2
    lr = lr0 * math.pow(drop, math.floor(cycle / cycle_drop))
    return lr


def run(model_path, 
        training_dir, 
        validation_dir, 
        log_dir, 
        learning_rate,
        batch_size,
        group_size,
        margin,
        input_shape,
        epochs_per_cycle,
        number_of_cycles):
    """Runs the training process.

    Args:
        model_path: Full path of the CNN model.
        training_dir: Path of the dataset root directory on which the 
            training will be executed.
        validation_dir: Path of the dataset root directory which will
            be used for evaluating the model at every epoch.
        log_dir: Path of the directory where the loss values will be saved.
        learning_rate: Float.
        batch_size: Integer. The size of the batch produced by the 
            data generator every step.
        group_size: Integer. The size of the group produced by the 
            data generator every step. To see what a group is, check 
            out the `utils.data_generator` module.
        margin: Float. Parameter of the triplet loss function.
        input_shape: Tuple. The shape of the model's input layer.
        epochs_per_cycle: Integer, the number of epochs done at each
            training cycle.
        number_of_cycles: Integer. The number of training cycles.
    """
    
    # Parameters

    lr0 = learning_rate
    model_name = os.path.split(model_path)[1]
    model_name = os.path.splitext(model_name)[0]
    model_dir = os.path.split(model_path)[0]
    save_name = f"{model_name}.h5"

    # If you want to start from a different snapshot, let's say 
    # `my_model_4.h5`, then change it to `cycle = 4`.
    # The value should be `cycle > 0`.
    cycle = 1

    while number_of_cycles > 0:

        # Training

        print(f"\nTrain cycle {cycle}")
        print("--------------")

        # Optionally use a learning rate decay
        # Just uncomment the proper one and comment the `lr = lr0` line
        # lr = time_decay(cycle, lr0)
        # lr = step_decay(cycle, lr0)
        lr = lr0

        # print(f"lr = {lr}")

        # You can change the frequency on which snapshots made here
        snapshot_point = 2

        if cycle % snapshot_point == 0:
            save_name = f"{model_name}_{cycle}.h5"
        else:
            save_name = f"{model_name}_{cycle - cycle % snapshot_point}.h5"

        if cycle > 1:
            if cycle % snapshot_point != 0:
                model_path = os.path.join(
                    model_dir, 
                    f"{model_name}_{cycle - cycle % snapshot_point}.h5")
            else:
                model_path = os.path.join(
                    model_dir, 
                    f"{model_name}_{cycle - snapshot_point}.h5")

        train.run(
            model_path, 
            training_dir, 
            validation_dir,
            batch_size=batch_size,
            group_size=group_size,
            epochs=epochs_per_cycle,
            learning_rate=lr,
            margin=margin,
            input_shape=input_shape,
            save_name=save_name,
            log_dir=log_dir,
            save=True)

        number_of_cycles -= 1
        cycle += 1

    print("-- Done! --")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("run_config_file")
    args = parser.parse_args()

    kwargs = config.get_session_args(args.run_config_file)

    run(**kwargs)
