"""Functions for creating mappings between images and their labels."""

import numpy as np
import os, csv


def create_csv(save_path, images_dir, image_map):
    """Creates an image - id mapping file in csv format.

    Maps the images found in `images_dir`.

    The format is based on the csv files that Kaggle.com provides for the whale
    datasets in
    https://www.kaggle.com/c/whale-categorization-playground/data
    and 
    https://www.kaggle.com/c/humpback-whale-identification/data.
    
    Args:
        save_path: Full path of the file.
        images_dir: The path of directory holding the dataset images.
        image_map: A dictionary which maps an image name to a whale id.
    """
    header = ["Image", "Id"]

    with open(save_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for image_name in os.listdir(images_dir):
            row = [image_name, image_map[image_name]]
            writer.writerow(row)


def create_fake_csv(save_path, images_dir):
    """Creates a meaningless csv for compatibility.

    It's handy for test datasets where the whales doesn't have any id
    provided.

    Args:
        save_path: Full path of the file.
        images_dir: The path of the directory holding the dataset images.
    """
    image_map = {}
    for image_name in os.listdir(images_dir):
        image_map.update({image_name: "new_whale"})
    create_csv(save_path, images_dir, image_map)


def read_csv(file_path, image_full_path=False):
    """Reads the image - id mapping csv and packs the data into a tuple.

    The returning format is a tuple of Numpy arrays: (x_data, y_data).
    The elements are
        'x_data' - the image file name (or full path) in shape: (size, 1),
        'y_data' - the corresponding whale id string in shape: (size, 1).

    This is the format Tensorflow 2.0 uses with its built-in datasets
    like CIFAR-10. See:
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/datasets/cifar10/load_data

    Args:
        file_path: Full path of the csv file.
        image_full_path: When set to 'true', it puts the full path of the
            image into the dictionary, otherwise just the image file name.
            Default is 'false'.

    Returns:
        A tuple of Numpy arrays: (x_data, y_data).
    """
    x_data = []
    y_data = []

    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader) # skip first row
    
        for row in reader:
            image_name = row[0]
            label = row[1]

            if image_full_path:
                head, tail = os.path.split(file_path)
                image_name = os.path.join(head, "images", image_name)

            x_data.append(image_name)
            y_data.append(label)

    x_data = np.stack(x_data, axis=0)
    y_data = np.stack(y_data, axis=0)

    x_data = np.reshape(x_data, (x_data.shape[0], 1))
    y_data = np.reshape(y_data, (y_data.shape[0], 1))

    return (x_data, y_data)


def get_image_map(data):
    """Creates a map from image to whale id.

    Args:
        data: A tuple of Numpy arrays: (x_data, y_data).

    Returns:
        A dictionary object.
    """
    x_train, y_train = data
    size = x_train.shape[0]

    image_map = {}

    for i in range(size):
        label = y_train[i][0]
        image = x_train[i][0]

        if image not in image_map:
            image_map.update({image: label})
        else:
            raise ValueError("Same image names!")

    return image_map


def get_label_map(data):
    """Creates a map from whale id to a list of image file names.
    
    Args:
        data: A tuple of Numpy arrays: (x_data, y_data).

    Returns:
        A dictionary object.
    """
    x_train, y_train = data
    size = x_train.shape[0]

    label_map = {}

    for i in range(size):
        label = y_train[i][0]
        
        if x_train.ndim == 2:
            image = x_train[i][0]
        else:
            image = x_train[i]

        if label not in label_map:
            label_map.update({label: [image]})
        else:
            label_map[label].append(image)

    return label_map


def csv2lblmap(file_path):
    """A shortcut to get a label map.
    """
    data = read_csv(file_path)
    return get_label_map(data)


def csv2imgmap(file_path):
    """A shortcut to get an image map.
    """
    data = read_csv(file_path)
    return get_image_map(data)