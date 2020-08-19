"""Handles configuration data.

Reads and parses data from configuration files and then prepares them
to pass on main modules such as `train` or `embed`.
"""
import os, configparser


_dir_config = configparser.ConfigParser()
_dir_config.read("config/dir_config.ini")


def read_run_config(run_config_file):
    """Reads the run configuration file.
    
    Args:
        config_file: Path of the config file.
    Returns:
        A dictionary object.
    """
    config = configparser.ConfigParser()
    config.read(run_config_file)

    model_name = config["model"]["name"]
    train_config = config["train"]

    input_shape = config["input_shape"]
    input_shape = (int(input_shape["width"]), 
                   int(input_shape["height"]), 
                   int(input_shape["channels"]))

    config = {
        "model_name":       model_name,
        "batch_size":       int(train_config["batch_size"]),
        "group_size":       int(train_config["group_size"]),
        "epochs":           int(train_config["epochs"]),
        "learning_rate":    float(train_config["learning_rate"]),
        "margin":           float(train_config["margin"]),
        "input_shape":      input_shape,
        "embed_save_dir":   config["dirs"]["embeddings"],
        "log_dir":          config["dirs"]["log"],
        "cycles":           int(config["session"]["cycles"])
    }

    return config
    

def get_train_args(run_config_file):
    run_config = read_run_config(run_config_file)

    train_args = {
        "model_path":       os.path.join(_dir_config["save_dirs"]["models"], 
                                         run_config["model_name"]),
        "training_dir":     _dir_config["dataset_dirs"]["train"],
        "validation_dir":   _dir_config["dataset_dirs"]["validation"],
        "batch_size":       run_config["batch_size"],
        "group_size":       run_config["group_size"],
        "epochs":           run_config["epochs"],
        "learning_rate":    run_config["learning_rate"],
        "margin":           run_config["margin"],
        "input_shape":      run_config["input_shape"],
        "save_name":        None,
        "log_dir":          os.path.join(
                                _dir_config["save_dirs"]["logs_root"], 
                                run_config["log_dir"])
    }

    return train_args


def get_embed_args(run_config_file):
    run_config = read_run_config(run_config_file)

    embed_args = {
        "model_path":   os.path.join(_dir_config["save_dirs"]["models"], 
                                     run_config["model_name"]),
        "save_dir":     os.path.join(
                            _dir_config["save_dirs"]["embeddings_root"], 
                            run_config["embed_save_dir"]),
        "dataset_dir":  _dir_config["dataset_dirs"]["train"],
        "input_shape":  run_config["input_shape"]
    }

    return embed_args


def get_evaluate_args(run_config_file):
    run_config = read_run_config(run_config_file)

    evaluate_args = {
        "model_path":   os.path.join(_dir_config["save_dirs"]["models"], 
                                     run_config["model_name"]),
        "dataset_dir":  _dir_config["dataset_dirs"]["validation"],
        "input_shape":  run_config["input_shape"],
        "batch_size":   run_config["batch_size"],
        "group_size":   run_config["group_size"],
        "log_dir":      os.path.join(_dir_config["save_dirs"]["logs_root"], 
                                     run_config["log_dir"])
    }

    return evaluate_args


def get_test_args(run_config_file):
    run_config = read_run_config(run_config_file)

    test_args = {
        "model_path":       os.path.join(_dir_config["save_dirs"]["models"], 
                                         run_config["model_name"]),
        "base_vectors_dir": os.path.join(
                                _dir_config["save_dirs"]["embeddings_root"], 
                                run_config["embed_save_dir"]),
        "dataset_dir":      _dir_config["dataset_dirs"]["validation"],
        "input_shape":      run_config["input_shape"]
    }

    return test_args


def get_session_args(run_config_file):
    run_config = read_run_config(run_config_file)

    session_args = {
        "model_path":       os.path.join(_dir_config["save_dirs"]["models"],
                                         run_config["model_name"]),
        "training_dir":     _dir_config["dataset_dirs"]["train"],
        "validation_dir":   _dir_config["dataset_dirs"]["validation"],
        "log_dir":          os.path.join(
                                _dir_config["save_dirs"]["logs_root"],
                                run_config["log_dir"]),
        "learning_rate":    run_config["learning_rate"],
        "batch_size":       run_config["batch_size"],
        "group_size":       run_config["group_size"],
        "margin":           run_config["margin"],
        "input_shape":      run_config["input_shape"],
        "epochs_per_cycle": run_config["epochs"],
        "number_of_cycles": run_config["cycles"]
    }

    return session_args