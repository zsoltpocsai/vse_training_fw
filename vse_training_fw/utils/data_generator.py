"""Provides Python data generators for training and evaluations."""

import tensorflow as tf
import numpy as np
import os, random, math
from utils import data_map


def process_image(image_path, image_size, channels):
    """Loads and converts image to a 3-dimensional Tensor.

    Also converts the pixel intensity values into 16-bit float in range
    of [0,1).

    Args:
        image_path: The image full path.
        image_size: A tuple: (width, height).
        channels: Number of channels the output image should have.

    Returns:
        A 3-dimensional Tensor: (width, height, channels).
    """
    if tf.rank(image_path) == 1:
        image_path = image_path[0]

    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels)
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tf.image.resize(image, image_size)

    return image


def for_train(data, batch_size, group_size, input_shape, preprocess=True):
    """Creates a Python generator of training data.

    In Tensorflow to start training the neural net model, a data generator
    is needed. There are a few options for that, on of them is a Python
    generator. The output of the generator must be a tuple `(inputs, targets)`,
    which packs up a batch size of data. `inputs` are the actual sample data
    (the image of the whale) and `targets` are the expected data (whale id).
    Both element is a Numpy array.

    For preparing the batch to yield, it also makes a selection of triplets,
    which consist an anchor and some positive and negative samples. The amount
    of these defined by the `group_size` and the `alpha` value. They tell
    how many samples to select for an anchor and what is the ratio of positive
    and negative ones.

    A batch consists groups of triplets, so the `batch_size` should be divisible
    by `group_size`.

    This generator itself never stops yielding new batch of datas. It will be
    the training process, which eventually stops.

    Args:
        data: A tuple `(x_data, y_data)`.
        batch_size: The amount of samples in a single batch.
        group_size: The size of a group consisting a single anchor sample
            and the rest filled up with positive and negative samples.
            The ratio of the positive and negatives is determined by the
            `alpha` value.
        input_shape: A tuple or a list `(width, height, channels)`.This must
            be the same as the input shape of the model. The images will be 
            converted into this shape.
        preprocess: If `False`, the image processing step will be skipped.
            The default value is `True`.

    Returns:
        A Python generator object.
    """
    group_count = math.ceil(batch_size / group_size)

    image_size = (input_shape[0], input_shape[1])
    channels = input_shape[2]
    
    alpha = 0.5
    positive_count = math.ceil(alpha * group_size)

    rnd = random.Random()
    rnd.seed()

    label_map = data_map.get_label_map(data)

    full_label_pool = list(label_map.keys())
    
    positive_label_pool = []

    while True:

        train_images = []
        train_labels = []

        for _ in range(group_count):

            #re-fill the positive label pool
            if len(positive_label_pool) < 1:
                positive_label_pool = list(
                    filter(lambda label: len(label_map[label]) >= 2, full_label_pool)
                )
                rnd.shuffle(positive_label_pool)

            #select positives
            positive_label = positive_label_pool.pop()
            positive_data_list = label_map[positive_label]
            
            if len(positive_data_list) < positive_count:
                positive_count = len(positive_data_list)

            positive_samples = rnd.sample(positive_data_list, positive_count)

            for positive_sample in positive_samples:

                if preprocess:
                    positive_sample = process_image(positive_sample, image_size, channels)

                train_images.append(positive_sample)
                train_labels.append(positive_label)

            #select negatives
            negative_count = group_size - positive_count

            for _ in range(negative_count):
                negative_label = rnd.choice(full_label_pool)
                negative_data_list = label_map[negative_label]
                negative_sample = rnd.choice(negative_data_list)

                if preprocess:
                    negative_sample = process_image(negative_sample, image_size, channels)

                train_images.append(negative_sample)
                train_labels.append(negative_label)

        train_images = np.stack(train_images, axis=0)
        train_labels = np.stack(train_labels, axis=0)

        yield (train_images, train_labels)


def for_validation(data, batch_size, group_size, input_shape, preprocess=True):
    """Creates a Python generator of data for validation.
    """
    return for_train(data, batch_size, group_size, input_shape, preprocess)


def for_test(data, input_shape, batch_size, preprocess=True):
    """Creates a Python generator of data for test.
    """
    return for_embed(data, input_shape, batch_size, preprocess)


def for_embed(data, input_shape, batch_size, preprocess=True):
    """Creates a Python generator for embedding data.
    
    For embedding there is no need to select triplets. The only thing it
    does is going through the given data and packs it into batches. 
    It will run out of samples and error will be thrown if the caller 
    method doesn't stop as well.

    Args:
        data: A tuple `(x_data, y_data)`.
        batch_size: The amount of samples in a single batch. Since in the
            embedding process it has no effect on the results, it can be 
            left on the default value 30.
        input_shape: A tuple or a list `(width, height, channels)`.This must
            be the same as the input shape of the model. The images will be 
            converted into this shape.
        preprocess: If `False`, the image processing step will be skipped.
            The default value is `True`.

    Returns:
        A Python generator object.
    """
    x_data = np.squeeze(data[0])
    x_data = list(x_data)
    data_size = len(x_data)

    steps = math.ceil(data_size / batch_size)
    remainder = data_size % batch_size
    last_step = steps - 1
    
    for step in range(steps):
        
        batch = []

        if step == last_step and remainder > 0:
            size = remainder
        else:
            size = batch_size

        for _ in range(size):
            image = x_data.pop(0)

            if preprocess:
                image_size = (input_shape[0], input_shape[1])
                channels = input_shape[2]
                image = process_image(image, image_size, channels)

            batch.append(image)

        batch = np.stack(batch, axis=0)

        yield (batch)


def enumerate_labels(data):
    """Subtitutes the `y_data` values with integer values.
    
    For the triplet loss implementation to work it is expected that the
    `y_data` is a Numpy array of integer values. This function assign
    a uniqe integer value to each distinct label (whale id) and replaces 
    them.

    Args:
        data: A tuple `(x_data, y_data)`.
    Returns:
        A tuple `(x_data, y_data)`.
    """
    x_data, y_data = data

    label_enum = {}
    number = 0

    y_data = np.squeeze(y_data)

    for label in y_data:
        if label not in label_enum:
            label_enum.update({label: number})
            number += 1

    y_data = list(map(lambda label: label_enum[label], y_data))

    y_data = np.stack(y_data, axis=0)
    y_data = np.reshape(y_data, (y_data.shape[0], 1))

    return (x_data, y_data)
