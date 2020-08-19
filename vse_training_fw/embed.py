"""For producing embedded vectors.

Can be run from the command line or used as a module by calling the 
`run` method.
"""

import argparse, os, math
import tensorflow as tf
import numpy as np

from utils import data_generator, data_map, config


def run(model_path,
        save_dir,
        dataset_dir,
        input_shape=(220, 220, 1),
        verbose=1):
    """Runs the vector-space embedding process.

    Args:
        model_path: Full path of the CNN model.
        save_dir: Path of directory where the embedded vectors and the
            corresponding metadatas will be saved.
        dataset_dir: Path of the dataset root directory.
        input_shape: Tuple. The shape of the model's input layer.
        verbose: If set to 1, prints some info to the command line.
            At 0, it will keep quiet. Default is 1.
    """

    # Loading the model

    model = tf.keras.models.load_model(model_path, compile=False)

    if verbose == 1:
        print("Model has been successfully loaded.")

    # Prepare the data generator
    # and calculate the steps

    labels_file = os.path.join(dataset_dir, "labels.csv")

    data_with_labels = data_map.read_csv(labels_file, image_full_path=True)
    data = data_generator.enumerate_labels(data_with_labels)

    batch_size = 30
    generator = data_generator.for_embed(data, input_shape, batch_size)

    data_size = len(data[0])    # data = (x_data, y_data)
    steps = math.ceil(data_size / batch_size)

    # Embedding

    if verbose == 1:
        print(f"Embedding {data_size} images...")

    embeddings = model.predict_generator(
        generator,
        steps,
        verbose=1
    )

    # Save the results

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # saving the vectors
    np.savetxt(os.path.join(save_dir, "vecs.tsv"), embeddings, delimiter='\t')

    # saving the labels
    meta_file = open(os.path.join(save_dir, "meta.tsv"), "w", encoding="utf-8")

    y_data = data_with_labels[1]
    y_data = np.squeeze(y_data)

    for label in y_data:
        meta_file.write(label + "\n")

    meta_file.close()

    if verbose == 1:
        print(f"Embedded vectors has been saved to '{save_dir}'")

    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_config_file")
    args = parser.parse_args()

    kwargs = config.get_embed_args(args.run_config_file)

    run(**kwargs)
