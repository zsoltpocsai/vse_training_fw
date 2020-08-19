"""Implements k-NN search in vector space."""

import os
import numpy as np
import tensorflow as tf


def _nearest_neighbours(value, array, k):
    """Calculates the `k` nearest neighbors of the given vector.
    Args:
        value: A Numpy array of vector coordinates.
        array: A Numpy array of the vector population in space.
        k: Integer. The number of nearest neighbor to look for.
    """
    distances = [np.linalg.norm(value-x) for x in array]

    nbr_indexes = []

    for i in range(k):
        min_element = min(distances)
        min_index = distances.index(min_element)
        nbr_indexes.append(min_index)
        distances.pop(min_index)

    return nbr_indexes


def get_predictions(base_vectors_dir, embeddings, k=5):
    """Packs up the `k` nearest neighbors for the given embedded vectors.

    For each given embedded vector in `embeddings` calculates the `k` closest
    vector in the vector space populated by the vectors in `base_vectors_dir`.
    Then creates a list of predictions with the same size as the `embeddings`,
    where each element is a list of the closest neighbors found.
    
    The index of embeddings and predictions elements corresponds to each other. 
    So the closest neighbors of `embeddings[i]` is placed to `predictions[i]`.
    
    The closest neighbors get the same labels as it is in `meta.tsv` file.

    Args:
        base_vectors_dir: Path of the directory containing the vectors which
            populates the vector space. It must have a `vecs.tsv` and
            a `meta.tsv` file in it.
        embeddings: A list or Numpy array of embedded vectors whose neighbors 
            are going to be looked for.
        k: Integer. The number of neighbors to look for. Default to 5.
    Returns:
        A list of array containing the labels of the closest neighbors.
    """
    vectors_file = base_vectors_dir + "/vecs.tsv"
    labels_file = base_vectors_dir + "/meta.tsv"

    base_vectors = np.loadtxt(vectors_file)

    labels_list = []
    f = open(labels_file, "r")
    line = f.readline()
    while line != "":
        labels_list.append(line.strip())
        line = f.readline()
    f.close()

    predictions = []

    for i in range(len(embeddings)):
        found_nbr_indexes = _nearest_neighbours(embeddings[i], base_vectors, k)
        predictions.append([labels_list[i] for i in found_nbr_indexes])

    return predictions
